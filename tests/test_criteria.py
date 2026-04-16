"""Unit tests for GeoTiffMap.satisfies() and MapRepository.get().

No file I/O — maps are built directly from numpy arrays.
"""

import numpy as np
import pytest

from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria
from pyroclast.io.data_repository import GeoTiffMap
from pyroclast.ABCs.repository import MapCriteria, RasterMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_habitat(code: str) -> GeoTiffMap:
    return GeoTiffMap(code=code, kind="habitat", data=np.zeros((4, 4), dtype=np.uint8))


def make_invasion() -> GeoTiffMap:
    return GeoTiffMap(code="invasion", kind="invasion", data=np.zeros((4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# GeoTiffMap.satisfies — HabitatCriteria
# ---------------------------------------------------------------------------

class TestHabitatCriteria:
    def test_habitat_matches_exact_code(self):
        assert make_habitat("9340").satisfies(HabitatCriteria(code="9340"))

    def test_habitat_no_match_wrong_code(self):
        assert not make_habitat("9340").satisfies(HabitatCriteria(code="92XX"))

    def test_habitat_matches_wildcard(self):
        assert make_habitat("9340").satisfies(HabitatCriteria(code=None))

    def test_invasion_does_not_match_habitat_criteria(self):
        assert not make_invasion().satisfies(HabitatCriteria(code=None))

    def test_invasion_does_not_match_specific_habitat(self):
        assert not make_invasion().satisfies(HabitatCriteria(code="9340"))


# ---------------------------------------------------------------------------
# GeoTiffMap.satisfies — InvasionCriteria
# ---------------------------------------------------------------------------

class TestInvasionCriteria:
    def test_invasion_matches_invasion_criteria(self):
        assert make_invasion().satisfies(InvasionCriteria())

    def test_habitat_does_not_match_invasion_criteria(self):
        assert not make_habitat("9340").satisfies(InvasionCriteria())


# ---------------------------------------------------------------------------
# MapRepository.get()
# ---------------------------------------------------------------------------

class _StubRepository(FileMapRepository):
    """Overrides matching() to return a fixed list without touching disk."""

    def __init__(self, maps: list[RasterMap]) -> None:
        self._maps = maps

    def matching(self, criteria: MapCriteria):
        return [m for m in self._maps if m.satisfies(criteria)]


class TestRepositoryGet:
    def test_get_returns_single_match(self):
        repo = _StubRepository([make_invasion()])
        result = repo.get(InvasionCriteria())
        assert result.kind == "invasion"

    def test_get_raises_when_no_match(self):
        repo = _StubRepository([make_habitat("9340")])
        with pytest.raises(ValueError, match="0"):
            repo.get(InvasionCriteria())

    def test_get_raises_when_multiple_matches(self):
        repo = _StubRepository([make_habitat("9340"), make_habitat("92XX")])
        with pytest.raises(ValueError, match="2"):
            repo.get(HabitatCriteria(code=None))
