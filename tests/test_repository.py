"""Integration tests for FileMapRepository against the real GeoTIFF files."""

import numpy as np
import pytest

from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria

DATA_DIR = "data"


@pytest.fixture(scope="module")
def repo() -> FileMapRepository:
    return FileMapRepository(DATA_DIR)


# ---------------------------------------------------------------------------
# matching() — cardinality
# ---------------------------------------------------------------------------

class TestMatching:
    def test_invasion_returns_one_map(self, repo):
        assert len(repo.matching(InvasionCriteria())) == 1

    def test_all_habitats_returns_three_maps(self, repo):
        assert len(repo.matching(HabitatCriteria())) == 3

    def test_specific_habitat_returns_one_map(self, repo):
        assert len(repo.matching(HabitatCriteria(code="9340"))) == 1

    def test_specific_habitat_code_is_correct(self, repo):
        (h,) = repo.matching(HabitatCriteria(code="9340"))
        assert h.code == "9340"

    def test_unknown_habitat_returns_empty(self, repo):
        assert repo.matching(HabitatCriteria(code="nonexistent")) == []


# ---------------------------------------------------------------------------
# Spatial consistency
# ---------------------------------------------------------------------------

class TestSpatialConsistency:
    def test_all_maps_have_same_shape(self, repo):
        invasion = repo.get(InvasionCriteria())
        for h in repo.matching(HabitatCriteria()):
            assert h.data.shape == invasion.data.shape, (
                f"Habitat '{h.code}' shape {h.data.shape} != invasion {invasion.data.shape}"
            )


# ---------------------------------------------------------------------------
# Data integrity — invasion map
# ---------------------------------------------------------------------------

class TestInvasionMapIntegrity:
    def test_no_nan_after_loading(self, repo):
        p = repo.get(InvasionCriteria()).data
        assert not np.any(np.isnan(p))

    def test_values_in_unit_interval(self, repo):
        p = repo.get(InvasionCriteria()).data
        assert p.min() >= 0.0
        assert p.max() <= 1.0

    def test_dtype_is_float32(self, repo):
        p = repo.get(InvasionCriteria()).data
        assert p.dtype == np.float32


# ---------------------------------------------------------------------------
# Data integrity — habitat maps
# ---------------------------------------------------------------------------

class TestHabitatMapIntegrity:
    def test_values_are_binary(self, repo):
        for h in repo.matching(HabitatCriteria()):
            unique = set(np.unique(h.data).tolist())
            assert unique <= {0, 1}, f"Habitat '{h.code}' has non-binary values: {unique}"

    def test_dtype_is_uint8(self, repo):
        for h in repo.matching(HabitatCriteria()):
            assert h.data.dtype == np.uint8, f"Habitat '{h.code}' dtype is {h.data.dtype}"

    def test_all_habitats_have_presence_cells(self, repo):
        for h in repo.matching(HabitatCriteria()):
            assert np.sum(h.data) > 0, f"Habitat '{h.code}' has no presence cells"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_data_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            FileMapRepository("nonexistent_dir").get(InvasionCriteria())
