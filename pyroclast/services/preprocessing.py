"""Preprocessing Service — orchestrates data loading, GPU computation, and caching.

This module is the *application service layer* for the preprocessing phase of
the pyroclast pipeline.  It sits between the domain/infrastructure layers and
the entry-points (CLI, notebooks, tests).

Architectural role
------------------
Following the Ports & Adapters pattern the service function
:func:`run_preprocessing_batch` depends **only on abstractions**:

* :class:`~pyroclast.ABCs.repository.MapRepository` — the data-loading Port.
* :class:`~pyroclast.ABCs.compute.IComputeAdapter` — the compute Port.

Concrete implementations (``FileMapRepository``, ``PyOpenCLAdapter``) are
injected by the caller, keeping the service testable with lightweight stubs.

Caching strategy
----------------
Results are persisted as ``.npy`` files in ``cache_dir`` using the naming
convention ``habitat_<code>.npy``.  Only the compacted probability vector
``p_vec`` is stored; ``n_cells`` is derived as ``len(p_vec)`` on load.

On each call the service:

1. Checks which habitats already have a cache file.
2. Loads cached habitats from disk (no GPU involved).
3. Forwards only the *uncached* habitats to the compute adapter.
4. Saves the new results to ``cache_dir``.
5. Returns the merged list in habitat-code-sorted order for determinism.

This design ensures that repeated runs are idempotent and incremental: adding
a new habitat to ``data/habitats/`` only triggers one new GPU call.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pyroclast.ABCs.compute import IComputeAdapter
from pyroclast.ABCs.repository import MapCriteria, MapRepository, RasterMap
from pyroclast.domain.models import CompactedHabitat
from pyroclast.io.data_repository import HabitatCriteria, InvasionCriteria

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, habitat_code: str) -> Path:
    """Return the canonical cache file path for a given habitat code.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory in which cache files are stored.
    habitat_code : str
        Unique identifier of the habitat (e.g. ``"9340"``).

    Returns
    -------
    pathlib.Path
        ``cache_dir / "habitat_<habitat_code>.npy"``
    """
    return cache_dir / f"habitat_{habitat_code}.npy"


def _load_from_cache(cache_dir: Path, habitat_code: str) -> CompactedHabitat:
    """Load a ``CompactedHabitat`` from a ``.npy`` cache file.

    The cache file stores only the compacted ``p_vec`` array; ``n_cells`` is
    derived from its length to avoid redundancy.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory containing the cache files.
    habitat_code : str
        Habitat code whose cache file should be loaded.

    Returns
    -------
    CompactedHabitat
        The reconstructed Value Object.

    Raises
    ------
    FileNotFoundError
        If the expected cache file does not exist.
    ValueError
        If the loaded array does not satisfy ``CompactedHabitat`` invariants.
    """
    path = _cache_path(cache_dir, habitat_code)
    p_vec: np.ndarray = np.load(path).astype(np.float32)
    return CompactedHabitat(
        habitat_code=habitat_code,
        n_cells=len(p_vec),
        p_vec=p_vec,
    )


def _save_to_cache(cache_dir: Path, result: CompactedHabitat) -> None:
    """Persist a ``CompactedHabitat``'s probability vector to a ``.npy`` file.

    Only ``p_vec`` is saved; ``habitat_code`` is encoded in the filename and
    ``n_cells`` is re-derived from ``len(p_vec)`` on load.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory in which the cache file will be written.  Must exist.
    result : CompactedHabitat
        The Value Object to cache.

    Raises
    ------
    OSError
        If the file cannot be written (e.g. permission error or full disk).
    """
    path = _cache_path(cache_dir, result.habitat_code)
    np.save(path, result.p_vec)
    logger.debug("Cache written: %s", path)


def run_preprocessing_batch(
    repo: MapRepository,
    compute: IComputeAdapter,
    criteria: MapCriteria,
    cache_dir: Path,
    force_recompute: bool = False,
) -> list[CompactedHabitat]:
    """Orchestrate the full preprocessing pipeline with intelligent caching.

    This is the primary entry-point for the preprocessing phase.  It loads
    raster data through the repository Port, delegates GPU computation to the
    compute Port, and manages a disk-based result cache to avoid redundant GPU
    work on subsequent runs.

    Pipeline steps
    --------------
    1. **Habitat discovery** — calls ``repo.matching(criteria)`` to obtain the
       full list of habitat :class:`~pyroclast.ABCs.repository.RasterMap`
       objects.
    2. **Invasion map retrieval** — calls ``repo.get(InvasionCriteria())`` to
       obtain the invasion-probability raster (always needed, even when all
       habitats are cached, to validate shapes).
    3. **Cache partition** — splits habitats into *cached* (a ``.npy`` file
       exists in ``cache_dir`` and ``force_recompute`` is ``False``) and
       *uncached*.
    4. **GPU batch** — forwards uncached habitats to
       ``compute.batch_preprocess(invasion, uncached_habitats)``.
    5. **Cache write** — saves each new result to ``cache_dir``.
    6. **Merge and return** — combines cached and newly computed results,
       sorted by ``habitat_code`` for determinism.

    Parameters
    ----------
    repo : MapRepository
        Repository Port used to load raster maps.  Typically a
        ``FileMapRepository`` in production and a stub in tests.
    compute : IComputeAdapter
        Compute Port used for GPU preprocessing.  Typically a
        ``PyOpenCLAdapter`` in production.  Not called when all habitats are
        already cached.
    criteria : MapCriteria
        Query predicate used to select habitat maps from the repository.
        Pass ``HabitatCriteria()`` to process all habitats, or
        ``HabitatCriteria(code="9340")`` to process a specific one.
    cache_dir : pathlib.Path
        Directory where ``.npy`` cache files are stored and read from.
        The directory **must already exist**; this function does not create it.
    force_recompute : bool, optional
        If ``True``, the cache is ignored and all habitats are sent to the
        compute adapter regardless of whether a cache file exists.  Results
        are still written to ``cache_dir``, overwriting any previous files.
        Defaults to ``False``.

    Returns
    -------
    list[CompactedHabitat]
        A list of :class:`~pyroclast.domain.models.CompactedHabitat` Value
        Objects, one per habitat matching ``criteria``, sorted by
        ``habitat_code``.  Returns an empty list if no habitats match.

    Raises
    ------
    ValueError
        If ``repo.get(InvasionCriteria())`` returns zero or more than one map,
        or if any habitat shape is inconsistent with the invasion map.
    FileNotFoundError
        If ``cache_dir`` does not exist.
    pyopencl.Error
        If a GPU error occurs inside ``compute.batch_preprocess``.

    Examples
    --------
    >>> from pathlib import Path
    >>> from pyroclast import FileMapRepository
    >>> from pyroclast.adapters import PyOpenCLAdapter
    >>> from pyroclast.io.data_repository import HabitatCriteria
    >>> from pyroclast.services import run_preprocessing_batch
    >>>
    >>> repo = FileMapRepository("data")
    >>> compute = PyOpenCLAdapter()
    >>> cache = Path("cache/")
    >>> cache.mkdir(exist_ok=True)
    >>>
    >>> habitats = run_preprocessing_batch(
    ...     repo=repo,
    ...     compute=compute,
    ...     criteria=HabitatCriteria(),
    ...     cache_dir=cache,
    ... )
    >>> for h in habitats:
    ...     print(h)
    """
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"cache_dir does not exist or is not a directory: {cache_dir}"
        )

    all_habitats: list[RasterMap] = list(repo.matching(criteria))
    if not all_habitats:
        logger.info("run_preprocessing_batch: no habitats match criteria %r.", criteria)
        return []

    invasion: RasterMap = repo.get(InvasionCriteria())
    logger.info(
        "run_preprocessing_batch: invasion map loaded, shape=%s.",
        invasion.data.shape,
    )

    cached_results: dict[str, CompactedHabitat] = {}
    uncached_habitats: list[RasterMap] = []

    for habitat in all_habitats:
        if not force_recompute and _cache_path(cache_dir, habitat.code).exists():
            logger.debug("Cache hit: habitat '%s'.", habitat.code)
            cached_results[habitat.code] = _load_from_cache(cache_dir, habitat.code)
        else:
            logger.debug("Cache miss: habitat '%s' queued for GPU.", habitat.code)
            uncached_habitats.append(habitat)

    logger.info(
        "run_preprocessing_batch: %d cached, %d to compute.",
        len(cached_results),
        len(uncached_habitats),
    )

    new_results: list[CompactedHabitat] = []
    if uncached_habitats:
        new_results = compute.batch_preprocess(invasion, uncached_habitats)
        for result in new_results:
            _save_to_cache(cache_dir, result)
            logger.info(
                "Computed and cached: habitat '%s' (%d active cells).",
                result.habitat_code,
                result.n_cells,
            )

    combined: dict[str, CompactedHabitat] = {
        **cached_results,
        **{r.habitat_code: r for r in new_results},
    }
    return sorted(combined.values(), key=lambda h: h.habitat_code)
