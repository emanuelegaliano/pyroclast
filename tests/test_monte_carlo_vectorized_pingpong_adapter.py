"""GPU integration tests for PyOpenCLMonteCarloVectorizedPingPongAdapter.

Same vectorized-RNG sampling as the tree-reduction vectorized kernel, but with
a ping-pong work-group reduction. Since the sampling is identical and integer
addition is associative, this kernel must be **bit-exact** with the
tree-reduction vectorized kernel at the same vec_width — that cross-check is
the main correctness gate here.
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_vectorized_adapter import (
    PyOpenCLMonteCarloVectorizedAdapter,
)
from pyroclast.adapters.opencl_mc_vectorized_pingpong_adapter import (
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
)
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig


WIDTHS = [2, 4, 8]


@pytest.fixture(scope="module", params=WIDTHS, ids=lambda w: f"w{w}")
def adapter(request):
    try:
        return PyOpenCLMonteCarloVectorizedPingPongAdapter(vec_width=request.param)
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str = "test") -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(n_runs=200_000, threshold=0.5, seed=42) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed)


# ---------------------------------------------------------------------------
# Deterministic corner cases
# ---------------------------------------------------------------------------

class TestDeterministicCases:
    def test_all_cells_certain_invasion(self, adapter):
        prob = adapter.run(_habitat([1.0] * 10), _config(threshold=0.5))
        assert prob == pytest.approx(1.0)

    def test_all_cells_zero_invasion(self, adapter):
        prob = adapter.run(_habitat([0.0] * 10), _config(threshold=0.0))
        assert prob == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Statistical convergence (incl. tail group, n_cells % W != 0)
# ---------------------------------------------------------------------------

class TestStatisticalConvergence:
    def test_single_cell_p05(self, adapter):
        prob = adapter.run(_habitat([0.5]), _config(threshold=0.0, seed=3, n_runs=100_000))
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_batched_run(self, adapter):
        prob = adapter.run_batched(
            _habitat([0.5]), _config(threshold=0.0, seed=3, n_runs=100_000), n_batches=2
        )
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_tail_mixed_converges(self, adapter):
        # 7 cells (< every width) → exercises the padded tail group.
        prob = adapter.run(_habitat([0.5] * 7), _config(threshold=0.0, n_runs=100_000, seed=9))
        assert prob == pytest.approx(1.0 - 0.5 ** 7, abs=0.01)


# ---------------------------------------------------------------------------
# Bit-exact equivalence with the tree-reduction vectorized kernel
# ---------------------------------------------------------------------------

class TestPingPongEquivalence:
    @pytest.mark.parametrize("w", WIDTHS, ids=lambda w: f"w{w}")
    def test_matches_tree_reduction(self, w):
        try:
            tree = PyOpenCLMonteCarloVectorizedAdapter(vec_width=w)
            pp = PyOpenCLMonteCarloVectorizedPingPongAdapter(vec_width=w)
        except Exception as exc:
            pytest.skip(f"OpenCL device unavailable: {exc}")

        hab = _habitat([0.1, 0.4, 0.7, 0.9, 0.05, 0.55, 0.3])
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.4, seed=1337)
        # Identical sampling + associative reduction ⇒ bit-exact totals.
        assert pp.run(hab, cfg) == tree.run(hab, cfg)

    @pytest.mark.parametrize("w", WIDTHS, ids=lambda w: f"w{w}")
    def test_matches_tree_reduction_batched(self, w):
        try:
            tree = PyOpenCLMonteCarloVectorizedAdapter(vec_width=w)
            pp = PyOpenCLMonteCarloVectorizedPingPongAdapter(vec_width=w)
        except Exception as exc:
            pytest.skip(f"OpenCL device unavailable: {exc}")

        hab = _habitat([0.3, 0.7])
        cfg = MonteCarloConfig(n_runs=100_000, threshold=0.0, seed=11)
        assert pp.run_batched(hab, cfg, n_batches=5) == tree.run_batched(hab, cfg, n_batches=5)
