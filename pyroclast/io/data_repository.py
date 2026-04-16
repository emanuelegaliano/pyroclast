import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio

from collections.abc import Sequence

from ..ABCs.repository import MapCriteria, RasterMap, MapRepositoryStrategy, MapRepository

_HABITAT_RE = re.compile(r"^cb_codice_(.+)\.tif$")


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HabitatCriteria(MapCriteria):
    """Matches habitat maps. code=None matches all habitats."""
    code: str | None = None


@dataclass(frozen=True)
class InvasionCriteria(MapCriteria):
    """Matches the invasion-probability map."""


# ---------------------------------------------------------------------------
# Domain object
# ---------------------------------------------------------------------------

class GeoTiffMap(RasterMap):

    def __init__(self, code: str, kind: str, data: np.ndarray) -> None:
        self._code = code
        self._kind = kind
        self._data = data

    @property
    def code(self) -> str:
        return self._code

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def data(self) -> np.ndarray:
        return self._data

    def satisfies(self, criteria: MapCriteria) -> bool:
        if isinstance(criteria, HabitatCriteria):
            if self._kind != "habitat":
                return False
            return criteria.code is None or criteria.code == self._code
        if isinstance(criteria, InvasionCriteria):
            return self._kind == "invasion"
        return False

    def __repr__(self) -> str:
        return f"GeoTiffMap(code={self._code!r}, kind={self._kind!r}, shape={self._data.shape})"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FileMapStrategy(MapRepositoryStrategy):
    """Loads maps from GeoTIFF files and caches them in memory."""

    def __init__(self, data_dir: Path) -> None:
        self._root = data_dir
        self._habitats_dir = data_dir / "habitats"

    @cached_property
    def _all_maps(self) -> list[GeoTiffMap]:
        maps: list[GeoTiffMap] = []

        invasion_candidates = sorted(self._root.glob("*.tif"))
        if not invasion_candidates:
            raise FileNotFoundError(f"No invasion map (.tif) found in {self._root}")

        with rasterio.open(invasion_candidates[0]) as src:
            p_data = src.read(1).astype(np.float32)
        np.nan_to_num(p_data, nan=0.0, copy=False)
        maps.append(GeoTiffMap(code="invasion", kind="invasion", data=p_data))

        for path in sorted(self._habitats_dir.glob("cb_codice_*.tif")):
            m = _HABITAT_RE.match(path.name)
            if not m:
                continue
            code = m.group(1)
            with rasterio.open(path) as src:
                h_data = src.read(1).astype(np.uint8)
            if h_data.shape != p_data.shape:
                raise ValueError(
                    f"Habitat '{code}' shape {h_data.shape} does not match "
                    f"invasion map shape {p_data.shape}"
                )
            maps.append(GeoTiffMap(code=code, kind="habitat", data=h_data))

        return maps

    def matching(self, criteria: MapCriteria) -> Sequence[GeoTiffMap]:
        return [m for m in self._all_maps if m.satisfies(criteria)]


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class FileMapRepository(MapRepository):
    """Concrete repository — reads from a directory of GeoTIFF files."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._strategy = FileMapStrategy(Path(data_dir))

    def matching(self, criteria: MapCriteria) -> Sequence[RasterMap]:
        return self._strategy.matching(criteria)
