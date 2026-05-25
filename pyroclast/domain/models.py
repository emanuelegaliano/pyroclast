"""Domain Value Objects for the pyroclast simulation.

This module is the innermost layer of the Ports & Adapters (Hexagonal)
architecture.  It has **no dependencies** on infrastructure, frameworks, or
I/O libraries — only on the Python standard library and NumPy.

Architectural role
------------------
Value Objects live in the *domain* ring.  They are created by Adapters after
GPU computation and consumed by the Service Layer, higher-level analytics, or
the Monte Carlo engine.  Their immutability guarantees referential transparency:
a ``CompactedHabitat`` always represents the same physical state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CompactedHabitat:
    """Immutable Value Object holding the GPU-preprocessed state of one habitat.

    After the Map kernel multiplies the invasion-probability raster by the
    habitat-presence mask, the result is a sparse float32 array where the
    vast majority of entries are zero (cells outside the habitat).
    ``CompactedHabitat`` stores only the *active* (non-zero) cells — the
    outcome of the stream-compaction step — together with the metadata needed
    to identify the habitat in subsequent pipeline stages.

    Architectural role
    ------------------
    This class is a *Value Object* in the DDD (Domain-Driven Design) sense: two instances with the
    same ``habitat_code`` and ``n_cells`` are considered structurally equal,
    regardless of the specific probability values they carry.  The probability
    vector ``p_vec`` is excluded from equality and hashing because:

    * NumPy ``__eq__`` returns an element-wise boolean array, which would
      break the dataclass-generated comparisons.
    * The semantic identity of a habitat snapshot is fully captured by its
      code and the count of active cells; ``p_vec`` is the payload, not the
      identity.

    Immutability
    ------------
    ``frozen=True`` prevents attribute reassignment after construction.  Note
    that the *contents* of ``p_vec`` (the underlying NumPy buffer) remain
    technically mutable; callers should treat the array as read-only.  A
    defensive copy can be obtained via ``habitat.p_vec.copy()``.

    Parameters
    ----------
    habitat_code : str
        Identifier of the habitat type (e.g. ``"9340"`` for *Quercus ilex*
        forests), as extracted from the GeoTIFF filename by
        ``FileMapStrategy``.  Must be a non-empty string.
    n_cells : int
        Number of active cells after stream compaction, i.e. ``len(p_vec)``.
        Must be non-negative and consistent with the length of ``p_vec``.
    p_vec : numpy.ndarray
        1-D ``float32`` array of length ``n_cells`` containing the per-cell
        invasion probability values for the active habitat cells
        (values in ``[0.0, 1.0]``).  Excluded from ``__eq__`` and
        ``__hash__``; see the note above.

    Raises
    ------
    ValueError
        If ``habitat_code`` is empty, ``n_cells`` is negative, ``p_vec`` is
        not 1-D, ``p_vec.dtype`` is not ``float32``, or
        ``len(p_vec) != n_cells``.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    >>> h = CompactedHabitat(habitat_code="9340", n_cells=3, p_vec=p)
    >>> h.total_probability
    1.5
    >>> h.mean_probability
    0.5
    """

    habitat_code: str
    n_cells: int
    p_vec: np.ndarray = field(compare=False, hash=False)

    def __post_init__(self) -> None:
        """Validate invariants immediately after dataclass construction.

        Raises
        ------
        ValueError
            On any structural inconsistency in the provided arguments.
        """
        if not self.habitat_code:
            raise ValueError("habitat_code must be a non-empty string.")
        if self.n_cells < 0:
            raise ValueError(
                f"n_cells must be non-negative, got {self.n_cells}."
            )
        if self.p_vec.ndim != 1:
            raise ValueError(
                f"p_vec must be 1-D, got shape {self.p_vec.shape}."
            )
        if self.p_vec.dtype != np.float32:
            raise ValueError(
                f"p_vec must have dtype float32, got {self.p_vec.dtype}."
            )
        if len(self.p_vec) != self.n_cells:
            raise ValueError(
                f"len(p_vec)={len(self.p_vec)} does not match n_cells={self.n_cells}."
            )

    @property
    def total_probability(self) -> float:
        """Sum of invasion probabilities across all active habitat cells.

        This is a proxy for the *expected number of active cells* that will be
        reached by a lava flow, under the assumption that cell events are
        independent.

        Returns
        -------
        float
            ``float(np.sum(self.p_vec))``.  Returns ``0.0`` when ``n_cells``
            is zero.
        """
        return float(np.sum(self.p_vec))

    @property
    def mean_probability(self) -> float:
        """Mean invasion probability across active habitat cells.

        Returns
        -------
        float
            ``float(np.mean(self.p_vec))``.  Returns ``0.0`` when ``n_cells``
            is zero to avoid division by zero.
        """
        if self.n_cells == 0:
            return 0.0
        return float(np.mean(self.p_vec))

    def __repr__(self) -> str:
        return (
            f"CompactedHabitat("
            f"habitat_code={self.habitat_code!r}, "
            f"n_cells={self.n_cells}, "
            f"mean_p={self.mean_probability:.4f})"
        )


@dataclass(frozen=True)
class SpatialHabitat:
    """Immutable Value Object holding a habitat in *2-D, uncompacted* form.

    Where :class:`CompactedHabitat` discards spatial layout (it stores only the
    non-zero ``p_vec`` values after stream compaction), ``SpatialHabitat`` keeps
    the full geographic footprint as a 2-D boolean *presence mask* aligned to
    the global probability map.  This is the input the **Map-Centric** Monte
    Carlo kernel needs: it lets the host build, for every map cell, a 64-bit
    bitmask of which habitats occupy that cell, so a single RNG draw per cell
    updates all overlapping habitats at once.

    Architectural role
    ------------------
    A *Value Object* in the domain ring (no infrastructure dependencies).  Two
    instances are structurally equal when their ``habitat_code`` and
    ``threshold`` match; ``presence_mask`` is excluded from equality and hashing
    for the same reasons ``CompactedHabitat.p_vec`` is (NumPy ``__eq__`` returns
    an array, and the payload is not part of the identity).

    Parameters
    ----------
    habitat_code : str
        Identifier of the habitat type.  Must be a non-empty string.
    presence_mask : numpy.ndarray
        2-D boolean array over the geographic grid; ``True`` marks a cell that
        belongs to this habitat.  Must share its shape with the global
        probability map passed to the adapter.  Excluded from ``__eq__`` and
        ``__hash__``.
    threshold : float
        Per-habitat critical invaded fraction :math:`\\theta \\in [0, 1]`.  A
        simulation destroys this habitat when its invaded fraction **strictly
        exceeds** ``threshold`` (matching the scalar baseline kernel).

    Raises
    ------
    ValueError
        If ``habitat_code`` is empty, ``presence_mask`` is not 2-D, or
        ``threshold`` is outside ``[0.0, 1.0]``.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[True, False], [False, True]])
    >>> h = SpatialHabitat(habitat_code="9340", presence_mask=mask, threshold=0.3)
    >>> h.n_cells
    2
    >>> h.shape
    (2, 2)
    """

    habitat_code: str
    presence_mask: np.ndarray = field(compare=False, hash=False)
    threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate invariants immediately after dataclass construction."""
        if not self.habitat_code:
            raise ValueError("habitat_code must be a non-empty string.")
        if self.presence_mask.ndim != 2:
            raise ValueError(
                f"presence_mask must be 2-D, got shape {self.presence_mask.shape}."
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {self.threshold}."
            )

    @property
    def n_cells(self) -> int:
        """Number of grid cells occupied by this habitat (count of ``True``)."""
        return int(np.count_nonzero(self.presence_mask))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(rows, cols)`` of the underlying 2-D presence mask."""
        return self.presence_mask.shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"SpatialHabitat("
            f"habitat_code={self.habitat_code!r}, "
            f"shape={self.shape}, "
            f"n_cells={self.n_cells}, "
            f"threshold={self.threshold:.3f})"
        )


@dataclass(frozen=True)
class BenchResult:
    """Immutable Value Object holding the results of a kernel benchmark run.

    Parameters
    ----------
    kernel_name : str
        Name of the OpenCL kernel that was benchmarked.
    shape : tuple[int, int]
        2-D grid shape used for the synthetic benchmark data.
    n_cells : int
        Total number of cells (``shape[0] * shape[1]``).
    n_runs : int
        Number of timed kernel launches (excluding warmup).
    mean_ms : float
        Mean kernel execution time in milliseconds.
    min_ms : float
        Minimum kernel execution time in milliseconds.
    bandwidth_gbs : float
        Effective memory bandwidth in GB/s, computed from the adapter's
        bytes-per-cell model and the mean execution time.
    """

    kernel_name: str
    shape: tuple[int, int]
    n_cells: int
    n_runs: int
    mean_ms: float
    min_ms: float
    bandwidth_gbs: float
    memory_mb: float


@dataclass(frozen=True)
class GridTopology:
    """Immutable Value Object representing the OpenCL execution grid.

    Parameters
    ----------
    gws : int or tuple[int, int]
        Global Work Size (total number of work-items).
    lws : int or tuple[int, int]
        Local Work Size (number of work-items per work-group).
    """

    gws: int | tuple[int, int]
    lws: int | tuple[int, int]

    def __post_init__(self) -> None:
        def _check_pos(v: int | tuple[int, ...], name: str) -> None:
            if isinstance(v, int):
                if v <= 0:
                    raise ValueError(f"{name} must be positive, got {v}.")
            else:
                for x in v:
                    if x <= 0:
                        raise ValueError(
                            f"{name} dimensions must be positive, got {v}."
                        )

        _check_pos(self.gws, "gws")
        _check_pos(self.lws, "lws")

        if isinstance(self.gws, int) != isinstance(self.lws, int):
            raise ValueError("gws and lws must be both int or both tuple.")

        if isinstance(self.gws, int):
            assert isinstance(self.lws, int)
            if self.gws % self.lws != 0:
                raise ValueError(
                    f"gws ({self.gws}) must be a multiple of lws ({self.lws})."
                )
        else:
            assert isinstance(self.lws, tuple)
            assert isinstance(self.gws, tuple)
            if len(self.gws) != len(self.lws):
                raise ValueError(
                    f"gws and lws tuples must have same length, got "
                    f"{len(self.gws)} and {len(self.lws)}."
                )
            for i, (g, l) in enumerate(zip(self.gws, self.lws)):
                if g % l != 0:
                    raise ValueError(
                        f"gws dimension {i} ({g}) must be a multiple of "
                        f"lws dimension {i} ({l})."
                    )


@dataclass(frozen=True)
class MonteCarloConfig:
    """Immutable Value Object carrying the parameters for a Monte Carlo run.

    Groups the parameters that govern a single Monte Carlo experiment: the
    number of independent runs, the critical invasion threshold, the seed,
    and the optional execution topology (GWS/LWS).

    Being a frozen dataclass, instances are safe to share across threads
    and to use as dictionary keys.

    Parameters
    ----------
    n_runs : int
        Number of independent Monte Carlo runs :math:`R`.  Must be positive.
        Values in the range 100 000–500 000 are typical for convergence.
    threshold : float
        Critical invaded fraction :math:`\\theta \\in [0, 1]`.  A run is
        counted as a *destruction event* when the fraction of invaded habitat
        cells **strictly exceeds** this value.
    seed : int
        Deterministic seed for the device-side RNG.  Must satisfy
        :math:`0 \\leq \\text{seed} < 2^{32}` so that it fits in an OpenCL
        ``uint`` argument.  The same ``(seed, n_runs, threshold)`` triple
        always produces the same result.
    topology : GridTopology or None, optional
        Execution grid configuration (GWS and LWS). If None, the adapter
        will choose a default topology.

    Raises
    ------
    ValueError
        If ``n_runs <= 0``, ``threshold`` is outside ``[0.0, 1.0]``, or
        ``seed`` is outside ``[0, 2³²−1]``.

    Examples
    --------
    >>> cfg = MonteCarloConfig(n_runs=100_000, threshold=0.5, seed=42)
    >>> cfg.n_runs
    100000
    >>> cfg.threshold
    0.5
    """

    n_runs: int
    threshold: float
    seed: int
    topology: GridTopology | None = None

    def __post_init__(self) -> None:
        if self.n_runs <= 0:
            raise ValueError(f"n_runs must be > 0, got {self.n_runs}.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {self.threshold}."
            )
        if not (0 <= self.seed < 2**32):
            raise ValueError(
                f"seed must be in [0, 2³²−1], got {self.seed}."
            )
