"""GPU integration tests for PyOpenCLMonteCarlo2DAdapter."""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_2d_stride_adapter import PyOpenCLMonteCarlo2DAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig, GridTopology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLMonteCarlo2DAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str = "test") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(
    n_runs: int = 200_000,
    threshold: float = 0.5,
    seed: int = 42,
    topology: GridTopology | None = None,
) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed, topology=topology)


# ---------------------------------------------------------------------------
# GridTopology 2D tests
# ---------------------------------------------------------------------------

class TestGridTopology2D:
    def test_valid_2d(self):
        topo = GridTopology(gws=(64, 16), lws=(32, 8))
        assert topo.gws == (64, 16)
        assert topo.lws == (32, 8)

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="multiple"):
            GridTopology(gws=(64, 16), lws=(32, 7))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            GridTopology(gws=(64, 16, 8), lws=(32, 8))

    def test_mixed_raises(self):
        with pytest.raises(ValueError, match="both int or both tuple"):
            GridTopology(gws=(64, 16), lws=256)


# ---------------------------------------------------------------------------
# Deterministic corner cases (2D Kernel)
# ---------------------------------------------------------------------------

class TestDeterministicCases2D:
    def test_all_cells_certain_invasion(self, adapter):
        hab = _habitat([1.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.5))
        assert prob == pytest.approx(1.0)

    def test_all_cells_zero_invasion(self, adapter):
        hab = _habitat([0.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.0))
        assert prob == pytest.approx(0.0)

    def test_custom_2d_topology(self, adapter):
        hab = _habitat([1.0] * 10)
        # Small grid to test stride
        topo = GridTopology(gws=(32, 8), lws=(32, 8))
        prob = adapter.run(hab, _config(threshold=0.5, n_runs=1000, topology=topo))
        assert prob == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Statistical convergence (2D Kernel)
# ---------------------------------------------------------------------------

class TestStatisticalConvergence2D:
    def test_single_cell_p05(self, adapter):
        hab = _habitat([0.5])
        prob = adapter.run(hab, _config(threshold=0.0, seed=3, n_runs=100_000))
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_batched_run(self, adapter):
        hab = _habitat([0.5])
        prob = adapter.run_batched(hab, _config(threshold=0.0, seed=3, n_runs=100_000), n_batches=2)
        assert prob == pytest.approx(0.5, abs=0.01)
