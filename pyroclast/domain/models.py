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
class MonteCarloConfig:
    """Immutable Value Object carrying the parameters for a Monte Carlo run.

    Groups the three scalar parameters that govern a single Monte Carlo
    experiment: the number of independent runs, the critical invasion
    threshold, and the seed for reproducible random number generation.

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
