"""Tests for the Map-Centric, 64-bit-bitmask Monte Carlo adapter.

The Map-Centric kernel draws its random numbers from a different stream layout
than the Habitat-Centric baseline (one draw per *map* cell vs. one per
*compacted habitat* cell), so it is **not** bit-exact with it. However, each
habitat's *marginal* destruction probability is provably identical in
distribution — sharing a cell's draw across overlapping habitats only correlates
the habitats with each other, it does not change any single habitat's marginal.
We therefore validate the Map-Centric adapter statistically against the standard
adapter, plus the batching (>64 habitats) and determinism guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.adapters.opencl_mc_map_centric_adapter import (
    PyOpenCLMapCentricAdapter,
)
from pyroclast.domain.models import (
    CompactedHabitat,
    MonteCarloConfig,
    SpatialHabitat,
)


@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLMapCentricAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


@pytest.fixture(scope="module")
def baseline():
    try:
        return PyOpenCLMonteCarloAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _spatial(mask2d: np.ndarray, code: str, threshold: float) -> SpatialHabitat:
    return SpatialHabitat(
        habitat_code=code, presence_mask=mask2d.astype(bool), threshold=threshold
    )


def _baseline_prob(
    baseline: PyOpenCLMonteCarloAdapter,
    p_map: np.ndarray,
    hab: SpatialHabitat,
    n_runs: int,
    seed: int,
) -> float:
    """Habitat-Centric reference probability for one SpatialHabitat.

    Compacts the map down to the habitat's occupied cells (exactly the data a
    ``CompactedHabitat`` would carry) and runs the standard scalar kernel.
    """
    idx = np.flatnonzero(hab.presence_mask.ravel())
    p_vec = np.ascontiguousarray(p_map.ravel()[idx], dtype=np.float32)
    ch = CompactedHabitat(
        habitat_code=hab.habitat_code, n_cells=int(idx.size), p_vec=p_vec
    )
    cfg = MonteCarloConfig(n_runs=n_runs, threshold=hab.threshold, seed=seed)
    return baseline.run(ch, cfg)


class TestEquivalenceWithHabitatCentric:
    """Per-habitat marginals must match the Habitat-Centric baseline."""

    def _scenario(self):
        """6x6 map; 3 partially-overlapping habitats, 1 isolated, 1 empty."""
        rng = np.random.default_rng(0)
        p_map = rng.uniform(0.2, 0.8, size=(6, 6)).astype(np.float32)

        def mask(rows, cols) -> np.ndarray:
            m = np.zeros((6, 6), dtype=bool)
            m[rows, cols] = True
            return m

        theta = 0.5
        habitats = [
            _spatial(mask(slice(0, 2), slice(0, 6)), "A", theta),  # rows 0-1
            _spatial(mask(slice(1, 3), slice(0, 6)), "B", theta),  # rows 1-2 (∩A row1)
            _spatial(mask(slice(2, 4), slice(0, 3)), "C", theta),  # ∩B row2 cols0-2
            _spatial(mask(slice(5, 6), slice(5, 6)), "ISO", theta),  # 1 cell, no overlap
            _spatial(np.zeros((6, 6), dtype=bool), "EMPTY", theta),  # empty
        ]
        return p_map, habitats

    def test_marginals_match_baseline(self, adapter, baseline):
        p_map, habitats = self._scenario()
        n_runs, seed = 200_000, 42
        cfg = MonteCarloConfig(n_runs=n_runs, threshold=0.0, seed=seed)

        mc = adapter.run_map(p_map, habitats, cfg)

        for hab in habitats:
            if hab.habitat_code == "EMPTY":
                assert mc["EMPTY"] == 0.0
                continue
            ref = _baseline_prob(baseline, p_map, hab, n_runs, seed)
            # Independent samples from differing stream layouts: ~6σ tolerance.
            assert mc[hab.habitat_code] == pytest.approx(ref, abs=0.01), (
                f"habitat {hab.habitat_code}: map-centric={mc[hab.habitat_code]:.4f} "
                f"baseline={ref:.4f}"
            )

    def test_overlapping_habitats_are_non_degenerate(self, adapter):
        """The shared-cell habitats must produce mid-range (not 0/1) probs,
        confirming the bitmask actually accumulates overlap correctly."""
        p_map, habitats = self._scenario()
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=7)
        mc = adapter.run_map(p_map, habitats, cfg)
        for code in ("A", "B", "C"):
            assert 0.05 < mc[code] < 0.95


class TestBatching:
    """More than 64 habitats must be split into chunks transparently."""

    def test_seventy_tiny_habitats(self, adapter):
        n_hab = 70  # → chunks of 64 + 6
        side = 10  # 100-cell map holds 70 distinct single-cell habitats
        p_map = np.full((side, side), 0.5, dtype=np.float32)

        habitats = []
        for h in range(n_hab):
            m = np.zeros((side, side), dtype=bool)
            m[h // side, h % side] = True  # one distinct cell each
            habitats.append(_spatial(m, f"H{h}", 0.0))

        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=123)
        mc = adapter.run_map(p_map, habitats, cfg)

        assert len(mc) == n_hab
        assert all(0.0 <= p <= 1.0 for p in mc.values())
        # Each habitat is a single p=0.5 cell with θ=0: P(invaded) = 0.5.
        for code, p in mc.items():
            assert p == pytest.approx(0.5, abs=0.02), f"{code}={p:.4f}"


class TestDeterminism:
    def test_same_seed_identical(self, adapter):
        p_map = np.full((4, 4), 0.5, dtype=np.float32)
        m1 = np.zeros((4, 4), dtype=bool)
        m1[0, :] = True
        m2 = np.zeros((4, 4), dtype=bool)
        m2[:, 0] = True
        habitats = [_spatial(m1, "row0", 0.3), _spatial(m2, "col0", 0.3)]
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=99)

        first = adapter.run_map(p_map, habitats, cfg)
        second = adapter.run_map(p_map, habitats, cfg)
        assert first == second
