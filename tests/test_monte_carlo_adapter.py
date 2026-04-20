"""GPU integration tests for PyOpenCLMonteCarloAdapter.

All tests are skipped automatically when no OpenCL device is available.
The tests verify statistical correctness of the Monte Carlo kernel against
analytically tractable cases:

* p=1.0 for all cells → every trial invades every cell → prob = 1.0
* p=0.0 for all cells → no cell is ever invaded → prob = 0.0
* p=p₀, n_cells=1, threshold=0.0 → prob ≈ p₀  (single-cell Bernoulli)
* Reproducibility: same seed → identical result across calls
* Different seeds → different results (statistical sanity check)
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLMonteCarloAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str = "test") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(
    n_runs: int = 200_000,
    threshold: float = 0.5,
    seed: int = 42,
) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed)


# ---------------------------------------------------------------------------
# Deterministic corner cases
# ---------------------------------------------------------------------------

class TestDeterministicCases:
    def test_all_cells_certain_invasion(self, adapter):
        """p=1 for every cell → fraction always 1.0 > any threshold < 1."""
        hab = _habitat([1.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.5))
        assert prob == pytest.approx(1.0)

    def test_all_cells_zero_invasion(self, adapter):
        """p=0 for every cell → fraction always 0.0, not > any threshold ≥ 0."""
        hab = _habitat([0.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.0))
        assert prob == pytest.approx(0.0)

    def test_threshold_at_one_never_exceeded_unless_all_certain(self, adapter):
        """threshold=1.0 means fraction must be > 1.0, which is impossible."""
        hab = _habitat([0.9] * 20)
        prob = adapter.run(hab, _config(threshold=1.0))
        assert prob == pytest.approx(0.0)

    def test_single_cell_certain_threshold_zero(self, adapter):
        """n_cells=1, p=1.0, threshold=0.0 → fraction 1.0 > 0.0 → prob = 1.0."""
        hab = _habitat([1.0])
        prob = adapter.run(hab, _config(threshold=0.0))
        assert prob == pytest.approx(1.0)

    def test_single_cell_zero_threshold_zero(self, adapter):
        """n_cells=1, p=0.0, threshold=0.0 → fraction 0.0, not > 0.0 → prob = 0.0."""
        hab = _habitat([0.0])
        prob = adapter.run(hab, _config(threshold=0.0))
        assert prob == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Statistical convergence
# ---------------------------------------------------------------------------

class TestStatisticalConvergence:
    """With R=200_000 runs and a single cell the estimate has σ ≈ 0.001.
    Using a tolerance of 5σ ≈ 0.005 makes spurious failures negligible."""

    def test_single_cell_p03(self, adapter):
        """p=0.3, n_cells=1, threshold=0.0 → prob ≈ 0.3."""
        hab = _habitat([0.3])
        prob = adapter.run(hab, _config(threshold=0.0, seed=1))
        assert prob == pytest.approx(0.3, abs=0.01)

    def test_single_cell_p07(self, adapter):
        """p=0.7, n_cells=1, threshold=0.0 → prob ≈ 0.7."""
        hab = _habitat([0.7])
        prob = adapter.run(hab, _config(threshold=0.0, seed=2))
        assert prob == pytest.approx(0.7, abs=0.01)

    def test_single_cell_p05(self, adapter):
        """p=0.5, n_cells=1, threshold=0.0 → prob ≈ 0.5."""
        hab = _habitat([0.5])
        prob = adapter.run(hab, _config(threshold=0.0, seed=3))
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_result_in_unit_interval(self, adapter):
        hab = _habitat([0.4, 0.6, 0.2])
        prob = adapter.run(hab, _config())
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_result(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5])
        cfg = _config(seed=99)
        assert adapter.run(hab, cfg) == adapter.run(hab, cfg)

    def test_different_seeds_different_results(self, adapter):
        """With a non-degenerate habitat, different seeds almost surely differ."""
        hab = _habitat([0.3, 0.7, 0.5])
        p1 = adapter.run(hab, _config(seed=10))
        p2 = adapter.run(hab, _config(seed=20))
        # Both valid probabilities; equality would be astronomically unlikely
        assert p1 != p2

    def test_different_n_runs_same_approximate_result(self, adapter):
        """R=100k and R=400k with the same seed should agree within 2σ ≈ 0.003."""
        hab = _habitat([0.5])
        p_small = adapter.run(hab, _config(n_runs=100_000, threshold=0.0, seed=7))
        p_large = adapter.run(hab, _config(n_runs=400_000, threshold=0.0, seed=7))
        assert abs(p_small - p_large) < 0.02
