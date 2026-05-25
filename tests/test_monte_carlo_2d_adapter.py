"""GPU integration tests for the 2-D Monte Carlo adapters.

The 2-D kernels parallelise the per-run cell scan across a ``cell_lanes`` axis
and close each run with two reductions (cells, then runs). Both the natural
(array-in-X) and transposed (runs-in-X) launch grids are exercised.

All tests are skipped automatically when no OpenCL device is available.

Correctness model
-----------------
* Deterministic corners (p=0/1, threshold=1) hold for any RNG stream layout.
* The interleaved cell partition uses the SAME stream layout as the vectorized
  kernel with ``VEC_WIDTH == cell_lanes``: it is therefore **bit-exact** with
  :class:`PyOpenCLMonteCarloVectorizedAdapter` of equal width, and with the two
  2-D variants of each other, but only **statistically** equal to the scalar
  1-D kernel.
* Changing ``run_lanes`` keeps the (run, cell) -> stream map unchanged, so it is
  bit-exact; changing ``cell_lanes`` changes the map, so it agrees statistically.
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_2d_adapter import PyOpenCLMonteCarlo2DAdapter
from pyroclast.adapters.opencl_mc_2d_transposed_adapter import (
    PyOpenCLMonteCarlo2DTransposedAdapter,
)
from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.adapters.opencl_mc_vectorized_adapter import (
    PyOpenCLMonteCarloVectorizedAdapter,
)
from pyroclast.domain.models import CompactedHabitat, GridTopology, MonteCarloConfig

_2D_FACTORIES = [
    PyOpenCLMonteCarlo2DAdapter,
    PyOpenCLMonteCarlo2DTransposedAdapter,
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=_2D_FACTORIES, ids=lambda f: f.__name__)
def adapter(request):
    try:
        return request.param()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _make(factory, **kwargs):
    """Construct an adapter or skip the test if no OpenCL device is present."""
    try:
        return factory(**kwargs)
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
    return MonteCarloConfig(
        n_runs=n_runs, threshold=threshold, seed=seed, topology=topology
    )


# ---------------------------------------------------------------------------
# Deterministic corner cases (layout-independent)
# ---------------------------------------------------------------------------

class TestDeterministicCases:
    def test_all_cells_certain_invasion(self, adapter):
        hab = _habitat([1.0] * 10)
        assert adapter.run(hab, _config(threshold=0.5)) == pytest.approx(1.0)

    def test_all_cells_zero_invasion(self, adapter):
        hab = _habitat([0.0] * 10)
        assert adapter.run(hab, _config(threshold=0.0)) == pytest.approx(0.0)

    def test_threshold_at_one_never_exceeded(self, adapter):
        hab = _habitat([0.9] * 20)
        assert adapter.run(hab, _config(threshold=1.0)) == pytest.approx(0.0)

    def test_single_cell_certain_threshold_zero(self, adapter):
        hab = _habitat([1.0])
        assert adapter.run(hab, _config(threshold=0.0)) == pytest.approx(1.0)

    def test_single_cell_zero_threshold_zero(self, adapter):
        hab = _habitat([0.0])
        assert adapter.run(hab, _config(threshold=0.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Statistical convergence (single-cell Bernoulli, sigma ~ 0.001 at R=200k)
# ---------------------------------------------------------------------------

class TestStatisticalConvergence:
    def test_single_cell_p03(self, adapter):
        hab = _habitat([0.3])
        assert adapter.run(hab, _config(threshold=0.0, seed=1)) == pytest.approx(
            0.3, abs=0.01
        )

    def test_single_cell_p07(self, adapter):
        hab = _habitat([0.7])
        assert adapter.run(hab, _config(threshold=0.0, seed=2)) == pytest.approx(
            0.7, abs=0.01
        )

    def test_result_in_unit_interval(self, adapter):
        hab = _habitat([0.4, 0.6, 0.2])
        assert 0.0 <= adapter.run(hab, _config()) <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_result(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5])
        cfg = _config(seed=99)
        assert adapter.run(hab, cfg) == adapter.run(hab, cfg)

    def test_different_seeds_different_results(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5, 0.1, 0.9, 0.45])
        p1 = adapter.run(hab, _config(threshold=0.4, seed=10))
        p2 = adapter.run(hab, _config(threshold=0.4, seed=20))
        assert p1 != p2

    def test_batched_matches_single(self, adapter):
        hab = _habitat([0.5])
        cfg = _config(n_runs=200_000, threshold=0.0, seed=7)
        assert adapter.run_batched(hab, cfg, n_batches=4) == pytest.approx(
            adapter.run(hab, cfg)
        )


# ---------------------------------------------------------------------------
# Grid-layout equivalences
# ---------------------------------------------------------------------------

class TestGridLayouts:
    def test_natural_and_transposed_bit_exact(self):
        """Same (cell_lanes, run_lanes): the two launch grids are bit-exact."""
        nat = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=32, run_lanes=8)
        tra = _make(
            PyOpenCLMonteCarlo2DTransposedAdapter, cell_lanes=32, run_lanes=8
        )
        hab = _habitat([0.1, 0.4, 0.7, 0.9, 0.05, 0.55])
        cfg = _config(n_runs=100_000, threshold=0.4, seed=1337)
        assert nat.run(hab, cfg) == tra.run(hab, cfg)

    def test_run_lanes_bit_exact(self):
        """Varying run_lanes keeps the (run, cell) stream map -> bit-exact."""
        a = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=64, run_lanes=2)
        b = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=64, run_lanes=8)
        hab = _habitat([0.3, 0.7, 0.5, 0.1, 0.9])
        cfg = _config(n_runs=100_000, threshold=0.4, seed=2024)
        assert a.run(hab, cfg) == b.run(hab, cfg)

    def test_cell_lanes_statistically_agree(self):
        """Varying cell_lanes changes the stream map -> statistical agreement."""
        a = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=16, run_lanes=4)
        b = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=64, run_lanes=4)
        hab = _habitat([0.3, 0.7, 0.5, 0.1, 0.9, 0.25])
        cfg = _config(n_runs=200_000, threshold=0.4, seed=5)
        assert a.run(hab, cfg) == pytest.approx(b.run(hab, cfg), abs=0.01)


# ---------------------------------------------------------------------------
# Cross-kernel consistency
# ---------------------------------------------------------------------------

class TestCrossKernelConsistency:
    def test_statistical_agreement_with_scalar_1d(self, adapter):
        """2-D agrees with the scalar 1-D kernel within Monte Carlo error."""
        ref = _make(PyOpenCLMonteCarloAdapter)
        hab = _habitat([0.3, 0.7, 0.5, 0.1, 0.9, 0.25, 0.6, 0.8])
        cfg = _config(n_runs=200_000, threshold=0.4, seed=123)
        assert adapter.run(hab, cfg) == pytest.approx(ref.run(hab, cfg), abs=0.01)

    @pytest.mark.parametrize("width", [2, 4, 8])
    def test_bit_exact_with_vectorized_same_width(self, width):
        """cell_lanes == VEC_WIDTH shares the stream layout -> bit-exact."""
        nat = _make(PyOpenCLMonteCarlo2DAdapter, cell_lanes=width, run_lanes=4)
        vec = _make(PyOpenCLMonteCarloVectorizedAdapter, vec_width=width)
        hab = _habitat([0.1, 0.4, 0.7, 0.9, 0.05, 0.55, 0.3, 0.65, 0.2])
        cfg = _config(n_runs=100_000, threshold=0.4, seed=777)
        assert nat.run(hab, cfg) == vec.run(hab, cfg)


# ---------------------------------------------------------------------------
# Work-group shape sweep
# ---------------------------------------------------------------------------

class TestWorkgroupShapes:
    @pytest.mark.parametrize(
        "cell_lanes,run_lanes", [(32, 8), (64, 4), (128, 2), (256, 1)]
    )
    def test_shapes_produce_valid_probability(self, cell_lanes, run_lanes):
        for factory in _2D_FACTORIES:
            a = _make(factory, cell_lanes=cell_lanes, run_lanes=run_lanes)
            hab = _habitat([0.4, 0.6, 0.2, 0.8, 0.1, 0.9])
            assert 0.0 <= a.run(hab, _config(seed=42)) <= 1.0
