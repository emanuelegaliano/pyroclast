"""GPU integration tests for PyOpenCLMonteCarloVectorizedAdapter.

The vectorized kernel uses MWC64X's vector RNG, so it is **not** bit-exact
with the scalar variants (different stream layout). It is therefore validated
statistically and on deterministic corner cases — not via the cross-variant
bit-exact test.
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_vectorized_adapter import (
    PyOpenCLMonteCarloVectorizedAdapter,
)
from pyroclast.domain.models import CompactedHabitat, GridTopology, MonteCarloConfig


WIDTHS = [2, 4, 8]


@pytest.fixture(scope="module", params=WIDTHS, ids=lambda w: f"w{w}")
def adapter(request):
    try:
        return PyOpenCLMonteCarloVectorizedAdapter(vec_width=request.param)
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
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_invalid_width_raises(self):
        with pytest.raises(ValueError, match="vec_width"):
            PyOpenCLMonteCarloVectorizedAdapter(vec_width=3)


# ---------------------------------------------------------------------------
# Deterministic corner cases (hold for any RNG layout)
# ---------------------------------------------------------------------------

class TestDeterministicCases:
    def test_all_cells_certain_invasion(self, adapter):
        hab = _habitat([1.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.5))
        assert prob == pytest.approx(1.0)

    def test_all_cells_zero_invasion(self, adapter):
        hab = _habitat([0.0] * 10)
        prob = adapter.run(hab, _config(threshold=0.0))
        # A draw is "invaded" only if x <= 0, i.e. x == 0 exactly (prob ~2^-24).
        # Tolerate the vanishingly rare hit rather than asserting exact zero.
        assert prob == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Statistical convergence
# ---------------------------------------------------------------------------

class TestStatisticalConvergence:
    def test_single_cell_p05(self, adapter):
        hab = _habitat([0.5])
        prob = adapter.run(hab, _config(threshold=0.0, seed=3, n_runs=100_000))
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_batched_run(self, adapter):
        hab = _habitat([0.5])
        prob = adapter.run_batched(
            hab, _config(threshold=0.0, seed=3, n_runs=100_000), n_batches=2
        )
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_tail_not_multiple_of_width(self, adapter):
        # n_cells = 10 is not a multiple of 4 or 8 → exercises the padded
        # tail group. Half the cells certain, half impossible: with theta just
        # below 0.5 the habitat is always destroyed.
        hab = _habitat([1.0] * 5 + [0.0] * 5)
        prob = adapter.run(hab, _config(threshold=0.49, n_runs=50_000, seed=7))
        assert prob == pytest.approx(1.0)

    def test_tail_mixed_converges(self, adapter):
        # 7 cells (prime, < every width): mean invaded fraction = 0.5, so with
        # theta = 0.0 destruction happens iff any cell is invaded ≈ always.
        hab = _habitat([0.5] * 7)
        prob = adapter.run(hab, _config(threshold=0.0, n_runs=100_000, seed=9))
        # P(no cell invaded) = 0.5^7 ≈ 0.0078, so P(destroyed) ≈ 0.992.
        assert prob == pytest.approx(1.0 - 0.5 ** 7, abs=0.01)


# ---------------------------------------------------------------------------
# Determinism (launch-independent, position-based seeding)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_config_identical(self, adapter):
        hab = _habitat([0.1, 0.4, 0.7, 0.9, 0.05, 0.55])
        cfg = _config(threshold=0.4, n_runs=100_000, seed=1337)
        assert adapter.run(hab, cfg) == adapter.run(hab, cfg)
