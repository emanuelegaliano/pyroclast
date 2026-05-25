"""Cross-variant consistency: every Monte Carlo adapter must return the
same probability for the same (habitat, config) tuple.

After the atomic-free refactor, the sampling kernels still produce
the same trial bits per ``(r, k)`` pair; only the work-group reduction
shape differs. Integer addition is associative, so the recursive reducer
yields a bit-exact total regardless of the variant or the number of
reduce passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.adapters.opencl_mc_pingpong_adapter import PyOpenCLMonteCarloPingPongAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig


# NOTE: PyOpenCLMonteCarloVectorizedAdapter is deliberately excluded — its
# vector RNG draws from a different stream layout, so it is not bit-exact with
# these variants. It is validated statistically in
# test_monte_carlo_vectorized_adapter.py instead.
_ADAPTER_FACTORIES = [
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
]


@pytest.fixture(scope="module")
def adapters():
    out = []
    for factory in _ADAPTER_FACTORIES:
        try:
            out.append(factory())
        except Exception as exc:
            pytest.skip(f"OpenCL device unavailable: {exc}")
    return out


def _habitat(p_values: list[float], code: str = "x") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


class TestCrossVariantConsistency:
    def test_single_cell_p05_seed_42(self, adapters):
        hab = _habitat([0.5])
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=42)
        results = [a.run(hab, cfg) for a in adapters]
        # All adapters draw from the same MWC64X streams; the sample
        # bit per (run, cell) is identical. The total integer count is
        # therefore identical too, regardless of the reduction shape.
        for r in results[1:]:
            assert r == results[0]

    def test_multi_cell_mixed_p_seed_1337(self, adapters):
        hab = _habitat([0.1, 0.4, 0.7, 0.9, 0.05, 0.55])
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.4, seed=1337)
        results = [a.run(hab, cfg) for a in adapters]
        for r in results[1:]:
            assert r == results[0]

    def test_batched_consistency(self, adapters):
        hab = _habitat([0.3, 0.7])
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=11)
        results = [a.run_batched(hab, cfg, n_batches=5) for a in adapters]
        for r in results[1:]:
            assert r == results[0]
