"""GPU integration tests for PyOpenCLMonteCarloContiguousAdapter.

Verifies statistical correctness, reproducibility, single-habitat equivalence,
and lack of cross-talk in the contiguous grid-stride MC kernel.
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_contiguous_adapter import PyOpenCLMonteCarloContiguousAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig


@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLMonteCarloContiguousAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str = "test") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(
    n_runs: int = 100_000,
    threshold: float = 0.5,
    seed: int = 42,
    topology = None,
) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed, topology=topology)


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

    def test_threshold_at_one_never_exceeded(self, adapter):
        """threshold=1.0 means fraction must be > 1.0, which is impossible."""
        hab = _habitat([0.9] * 20)
        prob = adapter.run(hab, _config(threshold=1.0))
        assert prob == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Statistical convergence
# ---------------------------------------------------------------------------

class TestStatisticalConvergence:
    """With R=100_000 runs and a single cell, standard deviation σ ≈ 0.0015.
    Using a tolerance of 0.015 makes spurious failures extremely unlikely."""

    def test_single_cell_p05(self, adapter):
        """p=0.5, n_cells=1, threshold=0.0 → prob ≈ 0.5."""
        hab = _habitat([0.5])
        prob = adapter.run(hab, _config(threshold=0.0, seed=42))
        assert prob == pytest.approx(0.5, abs=0.015)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_result(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5])
        cfg = _config(seed=99)
        assert adapter.run(hab, cfg) == adapter.run(hab, cfg)

    def test_different_seeds_different_results(self, adapter):
        hab = _habitat([0.3, 0.7, 0.5])
        p1 = adapter.run(hab, _config(seed=10))
        p2 = adapter.run(hab, _config(seed=20))
        assert p1 != p2


# ---------------------------------------------------------------------------
# Multi-Habitat and Cross-Talk Tests
# ---------------------------------------------------------------------------

class TestMultiHabitat:
    def test_empty_habitats(self, adapter):
        assert adapter.run_multi_habitats([], _config()) == []

    def test_single_habitat_equivalence(self, adapter):
        """Single habitat execution under run_multi_habitats matches run() exactly."""
        hab = _habitat([0.3, 0.6, 0.8, 0.1, 0.9, 0.4], "hab1")
        cfg = _config(n_runs=5000, seed=123)

        single_result = adapter.run(hab, cfg)
        multi_results = adapter.run_multi_habitats([hab], cfg)

        assert len(multi_results) == 1
        assert multi_results[0] == pytest.approx(single_result)

    def test_different_size_habitats_no_crosstalk(self, adapter):
        """Habitats with different sizes do not interfere/crosstalk."""
        hab1 = _habitat([1.0] * 5, "certain_destruction")
        hab2 = _habitat([0.0] * 15, "no_destruction")
        hab3 = _habitat([0.5] * 8, "probabilistic")

        cfg = _config(n_runs=5000, threshold=0.4, seed=789)

        multi_results = adapter.run_multi_habitats([hab1, hab2, hab3], cfg)

        assert len(multi_results) == 3
        assert multi_results[0] == pytest.approx(1.0)
        assert multi_results[1] == pytest.approx(0.0)
        assert 0.0 <= multi_results[2] <= 1.0
