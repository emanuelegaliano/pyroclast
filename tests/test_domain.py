"""Unit tests for CompactedHabitat — no GPU required."""

import numpy as np
import pytest

from pyroclast import CompactedHabitat


def _make(n: int = 4, code: str = "9340") -> CompactedHabitat:
    p = np.linspace(0.1, 0.9, n, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=n, p_vec=p)


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid(self):
        h = _make()
        assert h.habitat_code == "9340"
        assert h.n_cells == 4

    def test_empty_code_raises(self):
        with pytest.raises(ValueError, match="habitat_code"):
            CompactedHabitat(habitat_code="", n_cells=0, p_vec=np.array([], dtype=np.float32))

    def test_negative_n_cells_raises(self):
        with pytest.raises(ValueError, match="n_cells"):
            CompactedHabitat(habitat_code="X", n_cells=-1, p_vec=np.array([], dtype=np.float32))

    def test_2d_p_vec_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            CompactedHabitat(habitat_code="X", n_cells=4, p_vec=np.zeros((2, 2), dtype=np.float32))

    def test_wrong_dtype_raises(self):
        with pytest.raises(ValueError, match="float32"):
            CompactedHabitat(habitat_code="X", n_cells=3, p_vec=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_n_cells_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_cells"):
            CompactedHabitat(habitat_code="X", n_cells=99, p_vec=np.array([0.5], dtype=np.float32))

    def test_zero_cells_valid(self):
        h = CompactedHabitat(habitat_code="empty", n_cells=0, p_vec=np.array([], dtype=np.float32))
        assert h.n_cells == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_total_probability(self):
        p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        h = CompactedHabitat(habitat_code="X", n_cells=3, p_vec=p)
        assert abs(h.total_probability - 1.0) < 1e-5

    def test_mean_probability(self):
        p = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        h = CompactedHabitat(habitat_code="X", n_cells=3, p_vec=p)
        assert abs(h.mean_probability - 0.4) < 1e-5

    def test_mean_probability_zero_cells(self):
        h = CompactedHabitat(habitat_code="X", n_cells=0, p_vec=np.array([], dtype=np.float32))
        assert h.mean_probability == 0.0

    def test_total_probability_zero_cells(self):
        h = CompactedHabitat(habitat_code="X", n_cells=0, p_vec=np.array([], dtype=np.float32))
        assert h.total_probability == 0.0

    def test_repr_contains_code_and_n_cells(self):
        h = _make(code="9340")
        assert "9340" in repr(h)
        assert "4" in repr(h)

    def test_equality_by_code_and_n_cells(self):
        p1 = np.array([0.1, 0.9], dtype=np.float32)
        p2 = np.array([0.5, 0.5], dtype=np.float32)
        h1 = CompactedHabitat(habitat_code="A", n_cells=2, p_vec=p1)
        h2 = CompactedHabitat(habitat_code="A", n_cells=2, p_vec=p2)
        assert h1 == h2

    def test_inequality_different_code(self):
        h1 = _make(code="A")
        h2 = _make(code="B")
        assert h1 != h2
