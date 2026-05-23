"""GPU integration tests for PyOpenCLMonteCarlo2DTwoBarriersAdapter."""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_2d_two_barriers_adapter import PyOpenCLMonteCarlo2DTwoBarriersAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig, GridTopology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLMonteCarlo2DTwoBarriersAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str = "test") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(
    n_runs: int = 100_000,
    threshold: float = 0.5,
    seed: int = 42,
    topology: GridTopology | None = None,
) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed, topology=topology)


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
        prob = adapter.run(hab, _config(threshold=0.0, seed=3))
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_batched_run(self, adapter):
        hab = _habitat([0.5])
        prob = adapter.run_batched(hab, _config(threshold=0.0, seed=3), n_batches=2)
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_same_seed_identical_result(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5])
        cfg = _config(seed=99)
        assert adapter.run(hab, cfg) == adapter.run(hab, cfg)
