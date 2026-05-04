"""Concrete I/O implementations for the repository layer.

This module provides the infrastructure-facing side of the data-loading Port
defined in :mod:`pyroclast.ABCs.repository`.  It is the **only** place in the
codebase that reads GeoTIFF files from disk; all other modules depend solely on
the abstract :class:`~pyroclast.ABCs.repository.RasterMap` and
:class:`~pyroclast.ABCs.repository.MapRepository` interfaces.

Architectural role
------------------
Following the Ports & Adapters pattern, this module is a *primary adapter*
(driving adapter): it is the entry-point through which external data (GeoTIFF
files on disk) enters the application core.

The three concrete classes implement the three abstract interfaces defined in
:mod:`pyroclast.ABCs.repository`:

* :class:`GeoTiffMap` implements :class:`~pyroclast.ABCs.repository.RasterMap`.
* :class:`FileMapStrategy` implements
  :class:`~pyroclast.ABCs.repository.MapRepositoryStrategy`.
* :class:`FileMapRepository` implements
  :class:`~pyroclast.ABCs.repository.MapRepository`.

File layout conventions
-----------------------
The adapter expects the following directory structure::

    <data_dir>/
        <invasion_map>.tif          # exactly one .tif at root level
        habitats/
            cb_codice_<code>.tif    # one file per habitat type

The invasion map filename is not significant; the first ``.tif`` found at the
root level (alphabetically) is used.  Habitat filenames must match the pattern
``cb_codice_<code>.tif`` — the ``<code>`` segment becomes the
:attr:`GeoTiffMap.code` identifier (e.g. ``"9340"`` for *Quercus ilex*).

See also
--------
pyroclast.ABCs.repository : the abstract interfaces this module implements.
pyroclast.adapters.opencl_adapter : the compute-side adapter.
"""

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
    """Query predicate that selects habitat-presence maps.

    Pass this to :meth:`~pyroclast.ABCs.repository.MapRepository.matching` or
    :meth:`~pyroclast.ABCs.repository.MapRepository.get` to retrieve habitat
    rasters from the repository.

    Parameters
    ----------
    code : str or None, optional
        When provided, only the habitat whose :attr:`GeoTiffMap.code` matches
        this value is selected.  When ``None`` (the default), all habitat maps
        are selected regardless of their code.

    Examples
    --------
    >>> HabitatCriteria()            # matches all habitats
    HabitatCriteria(code=None)
    >>> HabitatCriteria(code="9340") # matches only habitat 9340
    HabitatCriteria(code='9340')
    """

    code: str | None = None


@dataclass(frozen=True)
class InvasionCriteria(MapCriteria):
    """Query predicate that selects the invasion-probability map.

    There is exactly one invasion map per dataset, so
    :meth:`~pyroclast.ABCs.repository.MapRepository.get` with this criteria
    always returns a single :class:`GeoTiffMap` of ``kind="invasion"``.

    Examples
    --------
    >>> repo.get(InvasionCriteria())   # returns the invasion raster
    """


# ---------------------------------------------------------------------------
# Domain object
# ---------------------------------------------------------------------------

class GeoTiffMap(RasterMap):
    """A raster layer loaded from a GeoTIFF file.

    Implements :class:`~pyroclast.ABCs.repository.RasterMap`.  Instances are
    created by :class:`FileMapStrategy` and are never constructed directly by
    application code.

    Each instance holds the full raster array in memory as a NumPy array.
    Habitat maps are stored as ``uint8`` (0/1 presence mask); the invasion map
    is stored as ``float32`` (probability in ``[0.0, 1.0]``, with NaN replaced
    by ``0.0`` at load time).

    Parameters
    ----------
    code : str
        Unique identifier for this map.  For habitat maps this is the substring
        extracted from the filename pattern ``cb_codice_<code>.tif``
        (e.g. ``"9340"``).  For the invasion map the fixed value ``"invasion"``
        is used.
    kind : str
        Category of the map.  Either ``"habitat"`` or ``"invasion"``.  Used by
        :meth:`satisfies` to dispatch criteria matching.
    data : numpy.ndarray
        The raster data as a 2-D NumPy array.  ``dtype=uint8`` for habitat
        maps, ``dtype=float32`` for the invasion map.

    Examples
    --------
    >>> import numpy as np
    >>> m = GeoTiffMap(code="9340", kind="habitat", data=np.zeros((4, 4), dtype=np.uint8))
    >>> m.code
    '9340'
    >>> m.kind
    'habitat'
    >>> m.satisfies(HabitatCriteria(code="9340"))
    True
    """

    def __init__(self, code: str, kind: str, data: np.ndarray) -> None:
        self._code = code
        self._kind = kind
        self._data = data

    @property
    def code(self) -> str:
        """Unique identifier of this map (e.g. ``"9340"`` or ``"invasion"``)."""
        return self._code

    @property
    def kind(self) -> str:
        """Category of this map: ``"habitat"`` or ``"invasion"``."""
        return self._kind

    @property
    def data(self) -> np.ndarray:
        """The raster data as a 2-D NumPy array.

        Returns
        -------
        numpy.ndarray
            ``dtype=uint8`` for habitat maps (0/1 presence mask),
            ``dtype=float32`` for the invasion map (probability values in
            ``[0.0, 1.0]``).
        """
        return self._data

    def satisfies(self, criteria: MapCriteria) -> bool:
        """Return whether this map matches the given query predicate.

        Dispatches on the concrete type of ``criteria``:

        * :class:`HabitatCriteria` — returns ``True`` only when
          ``self.kind == "habitat"`` and either ``criteria.code is None`` or
          ``criteria.code == self.code``.
        * :class:`InvasionCriteria` — returns ``True`` only when
          ``self.kind == "invasion"``.
        * Any other ``MapCriteria`` subclass — returns ``False``.

        Parameters
        ----------
        criteria : MapCriteria
            The query predicate to evaluate against this map.

        Returns
        -------
        bool
            ``True`` if the map matches the criteria, ``False`` otherwise.
        """
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
    """Loads all maps from GeoTIFF files and keeps them in memory.

    Implements :class:`~pyroclast.ABCs.repository.MapRepositoryStrategy`.
    On first access, reads all ``.tif`` files under ``data_dir`` according to
    the layout conventions described in the module docstring and caches the
    result for the lifetime of the instance.  Subsequent calls to
    :meth:`matching` hit the in-memory cache with no further disk I/O.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root data directory.  Must contain a ``habitats/`` subdirectory with
        files named ``cb_codice_<code>.tif``.  If ``invasion_map`` is not
        given, exactly one ``.tif`` file must exist at the top level.
    invasion_map : pathlib.Path or None, optional
        Explicit path to the invasion-probability GeoTIFF.  When provided,
        auto-discovery of ``.tif`` files in ``data_dir`` is skipped.  Use
        this when other ``.tif`` files (e.g. a DEM) share the same directory.

    Raises
    ------
    FileNotFoundError
        If no invasion map can be found (neither explicit nor via discovery).
    ValueError
        If any habitat raster shape does not match the invasion map shape.
    """

    def __init__(self, data_dir: Path, invasion_map: Path | None = None) -> None:
        self._root = data_dir
        self._habitats_dir = data_dir / "habitats"
        self._invasion_map = invasion_map

    @cached_property
    def _all_maps(self) -> list[GeoTiffMap]:
        """Load and cache all maps from disk (called at most once)."""
        maps: list[GeoTiffMap] = []

        if self._invasion_map is not None:
            invasion_path = self._invasion_map
            if not invasion_path.is_file():
                raise FileNotFoundError(f"Invasion map not found: {invasion_path}")
        else:
            invasion_candidates = sorted(self._root.glob("*.tif"))
            if not invasion_candidates:
                raise FileNotFoundError(f"No invasion map (.tif) found in {self._root}")
            invasion_path = invasion_candidates[0]

        with rasterio.open(invasion_path) as src:
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
        """Return all maps that satisfy the given criteria.

        Parameters
        ----------
        criteria : MapCriteria
            The query predicate used to filter maps.

        Returns
        -------
        list[GeoTiffMap]
            All maps in the dataset for which
            :meth:`GeoTiffMap.satisfies` returns ``True``.  May be empty.
        """
        return [m for m in self._all_maps if m.satisfies(criteria)]


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class FileMapRepository(MapRepository):
    """Repository that reads raster maps from a directory of GeoTIFF files.

    Implements :class:`~pyroclast.ABCs.repository.MapRepository`.  This is the
    primary entry-point for the data-loading layer in production use.  It
    delegates all file I/O and in-memory caching to a :class:`FileMapStrategy`
    instance.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Path to the root data directory.  Defaults to ``"data"``.  The
        directory must follow the layout described in the module docstring.

    Examples
    --------
    >>> from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria
    >>> repo = FileMapRepository("data")
    >>> invasion = repo.get(InvasionCriteria())
    >>> habitats = repo.matching(HabitatCriteria())
    """

    def __init__(self, data_dir: str | Path = "data", invasion_map: str | Path | None = None) -> None:
        self._strategy = FileMapStrategy(
            Path(data_dir),
            invasion_map=Path(invasion_map) if invasion_map else None,
        )

    def matching(self, criteria: MapCriteria) -> Sequence[RasterMap]:
        """Return all maps matching the given criteria.

        Delegates to :meth:`FileMapStrategy.matching`.  The full dataset is
        loaded from disk on the first call and cached for subsequent ones.

        Parameters
        ----------
        criteria : MapCriteria
            Query predicate — typically :class:`HabitatCriteria` or
            :class:`InvasionCriteria`.

        Returns
        -------
        Sequence[RasterMap]
            All maps for which :meth:`GeoTiffMap.satisfies` returns ``True``.
            Returns an empty sequence if no maps match.
        """
        return self._strategy.matching(criteria)
