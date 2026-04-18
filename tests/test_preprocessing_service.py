"""Unit tests for run_preprocessing_batch — stubs for repo and compute."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyroclast import CompactedHabitat, IComputeAdapter, run_preprocessing_batch
from pyroclast.ABCs.repository import MapCriteria, RasterMap
from pyroclast.io.data_repository import FileMapRepository, GeoTiffMap, HabitatCriteria, InvasionCriteria


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _invasion() -> GeoTiffMap:
    return GeoTiffMap(code="invasion", kind="invasion", data=np.zeros((4, 4), dtype=np.float32))


def _habitat(code: str) -> GeoTiffMap:
    return GeoTiffMap(code=code, kind="habitat", data=np.zeros((4, 4), dtype=np.uint8))


class _StubRepo(FileMapRepository):
    def __init__(self, maps: list[RasterMap]) -> None:
        self._maps = maps

    def matching(self, criteria: MapCriteria) -> list[RasterMap]:
        return [m for m in self._maps if m.satisfies(criteria)]


class _StubCompute(IComputeAdapter):
    def __init__(self) -> None:
        self.call_count = 0
        self.last_habitats: list[RasterMap] = []

    def batch_preprocess(self, invasion_map, habitats):
        self.call_count += 1
        self.last_habitats = list(habitats)
        return [
            CompactedHabitat(habitat_code=h.code, n_cells=0, p_vec=np.array([], dtype=np.float32))
            for h in habitats
        ]


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPreprocessingBatch:
    def test_empty_habitats_returns_empty(self, tmp):
        repo = _StubRepo([_invasion()])
        result = run_preprocessing_batch(repo, _StubCompute(), HabitatCriteria(), tmp)
        assert result == []

    def test_compute_called_once_for_uncached(self, tmp):
        repo = _StubRepo([_invasion(), _habitat("A"), _habitat("B")])
        compute = _StubCompute()
        run_preprocessing_batch(repo, compute, HabitatCriteria(), tmp)
        assert compute.call_count == 1

    def test_compute_not_called_when_all_cached(self, tmp):
        repo = _StubRepo([_invasion(), _habitat("A")])
        compute = _StubCompute()
        run_preprocessing_batch(repo, compute, HabitatCriteria(), tmp)
        run_preprocessing_batch(repo, compute, HabitatCriteria(), tmp)
        assert compute.call_count == 1

    def test_only_uncached_passed_to_compute(self, tmp):
        repo = _StubRepo([_invasion(), _habitat("A"), _habitat("B")])
        # Pre-cache only A
        run_preprocessing_batch(repo, _StubCompute(), HabitatCriteria(code="A"), tmp)
        # Now run for both: only B should reach compute
        compute = _StubCompute()
        run_preprocessing_batch(repo, compute, HabitatCriteria(), tmp)
        assert [h.code for h in compute.last_habitats] == ["B"]

    def test_results_sorted_by_habitat_code(self, tmp):
        repo = _StubRepo([_invasion(), _habitat("Z"), _habitat("A"), _habitat("M")])
        result = run_preprocessing_batch(repo, _StubCompute(), HabitatCriteria(), tmp)
        assert [r.habitat_code for r in result] == ["A", "M", "Z"]

    def test_cache_dir_missing_raises(self):
        repo = _StubRepo([_invasion(), _habitat("A")])
        with pytest.raises(FileNotFoundError):
            run_preprocessing_batch(repo, _StubCompute(), HabitatCriteria(), Path("/nonexistent/xyz"))

    def test_cached_p_vec_round_trips(self, tmp):
        p = np.array([0.3, 0.7], dtype=np.float32)

        class _FixedCompute(IComputeAdapter):
            def batch_preprocess(self, invasion_map, habitats):
                return [CompactedHabitat(habitat_code=habitats[0].code, n_cells=2, p_vec=p.copy())]

        repo = _StubRepo([_invasion(), _habitat("A")])
        run_preprocessing_batch(repo, _FixedCompute(), HabitatCriteria(), tmp)
        result = run_preprocessing_batch(repo, _FixedCompute(), HabitatCriteria(), tmp)
        np.testing.assert_array_almost_equal(result[0].p_vec, p)
