"""Tests for PyOpenCLAdapter — requires an OpenCL-capable device."""

import numpy as np
import pytest

from pyroclast import CompactedHabitat, PyOpenCLAdapter
from pyroclast.io.data_repository import GeoTiffMap


@pytest.fixture(scope="module")
def adapter():
    try:
        return PyOpenCLAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _invasion(shape=(8, 8)) -> GeoTiffMap:
    rng = np.random.default_rng(0)
    return GeoTiffMap(code="invasion", kind="invasion", data=rng.random(shape).astype(np.float32))


def _habitat(code="H", shape=(8, 8)) -> GeoTiffMap:
    rng = np.random.default_rng(1)
    return GeoTiffMap(code=code, kind="habitat", data=rng.integers(0, 2, size=shape, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestBasic:
    def test_empty_habitats_returns_empty(self, adapter):
        assert adapter.batch_preprocess(_invasion(), []) == []

    def test_one_result_per_habitat(self, adapter):
        results = adapter.batch_preprocess(_invasion(), [_habitat("A"), _habitat("B")])
        assert len(results) == 2

    def test_result_order_matches_input(self, adapter):
        habitats = [_habitat("A"), _habitat("B"), _habitat("C")]
        codes = [r.habitat_code for r in adapter.batch_preprocess(_invasion(), habitats)]
        assert codes == ["A", "B", "C"]

    def test_returns_compacted_habitat(self, adapter):
        (result,) = adapter.batch_preprocess(_invasion(), [_habitat()])
        assert isinstance(result, CompactedHabitat)

    def test_p_vec_dtype_float32(self, adapter):
        (result,) = adapter.batch_preprocess(_invasion(), [_habitat()])
        assert result.p_vec.dtype == np.float32

    def test_n_cells_matches_p_vec_length(self, adapter):
        (result,) = adapter.batch_preprocess(_invasion(), [_habitat()])
        assert result.n_cells == len(result.p_vec)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_zero_habitat_mask_yields_empty_p_vec(self, adapter):
        h_data = np.zeros((8, 8), dtype=np.uint8)
        habitat = GeoTiffMap(code="empty", kind="habitat", data=h_data)
        (result,) = adapter.batch_preprocess(_invasion(), [habitat])
        assert result.n_cells == 0

    def test_all_ones_mask_keeps_all_nonzero_cells(self, adapter):
        rng = np.random.default_rng(42)
        p_data = rng.random((8, 8)).astype(np.float32)
        invasion = GeoTiffMap(code="invasion", kind="invasion", data=p_data)
        h_data = np.ones((8, 8), dtype=np.uint8)
        habitat = GeoTiffMap(code="full", kind="habitat", data=h_data)
        (result,) = adapter.batch_preprocess(invasion, [habitat])
        expected = int(np.count_nonzero(p_data))
        assert result.n_cells == expected

    def test_output_values_bounded_by_invasion(self, adapter):
        rng = np.random.default_rng(7)
        p_data = rng.random((16, 16)).astype(np.float32)
        invasion = GeoTiffMap(code="invasion", kind="invasion", data=p_data)
        habitat = _habitat(shape=(16, 16))
        (result,) = adapter.batch_preprocess(invasion, [habitat])
        assert result.p_vec.max() <= p_data.max() + 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_shape_mismatch_raises(self, adapter):
        invasion = _invasion(shape=(8, 8))
        habitat = _habitat(shape=(4, 4))
        with pytest.raises(ValueError, match="shape"):
            adapter.batch_preprocess(invasion, [habitat])
