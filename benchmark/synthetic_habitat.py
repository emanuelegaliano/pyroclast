"""Synthetic habitat generator for benchmarks and statistical tests.

Produces a :class:`~pyroclast.domain.models.CompactedHabitat` whose invasion
probabilities are drawn from a Beta distribution, giving a non-degenerate
(neither 0 nor 1) destruction probability under the chosen threshold.

The design rationale is:

* **Reproducibility** — a fixed RNG seed produces the same ``p_vec`` every
  time, so benchmarks and statistical tests are independent of the real data
  files and fully reproducible.
* **Non-degeneracy** — by choosing ``threshold`` close to the analytical mean
  of the Beta distribution we ensure ``0 < p̂ < 1``, which is required for both
  the Z-test and the chi-square test to be defined.
* **Realistic scale** — ``n_cells = 50 000`` is representative of the smaller
  real habitats (habitat 4090 has ~154 k cells) while keeping GPU kernel
  launches fast enough for repeated statistical experiments.

Default parameters
------------------
* Beta(α=2, β=5) → mean = 2/7 ≈ 0.286, mode ≈ 0.167.
* threshold = 0.28 ≈ mean − ε → expected P(destruction) ≈ 0.55.

Usage
-----
::

    from benchmark.synthetic_habitat import make_synthetic_habitat

    habitat, threshold = make_synthetic_habitat()
    # habitat is a CompactedHabitat ready to pass to any MC adapter
"""

from __future__ import annotations

import numpy as np

from pyroclast.domain.models import CompactedHabitat, SpatialHabitat


def make_synthetic_habitat(
    n_cells: int = 50_000,
    beta_a: float = 2.0,
    beta_b: float = 5.0,
    threshold: float = 0.28,
    seed: int = 42,
    habitat_code: str = "SYNTHETIC",
) -> tuple[CompactedHabitat, float]:
    """Create a synthetic habitat with Beta-distributed invasion probabilities.

    Parameters
    ----------
    n_cells : int
        Number of active habitat cells.  Default: 50 000.
    beta_a, beta_b : float
        Shape parameters of the Beta distribution used to sample ``p_vec``.
        Default: α=2, β=5 (mean ≈ 0.286, std ≈ 0.149).
    threshold : float
        Destruction threshold θ passed to the Monte Carlo kernel.  Should be
        close to (but slightly below) the Beta mean so that the expected
        destruction probability is ~0.5 — the most sensitive operating point
        for the statistical tests.  Default: 0.28.
    seed : int
        NumPy RNG seed for reproducibility.  Default: 42.
    habitat_code : str
        Identifier embedded in the returned ``CompactedHabitat``.

    Returns
    -------
    habitat : CompactedHabitat
        Ready-to-use synthetic habitat.
    threshold : float
        The threshold value to use with the Monte Carlo config (returned for
        convenience so callers don't have to repeat the same constant).

    Notes
    -----
    The expected fraction of invaded cells in one simulation is
    ``mean(p_vec) ≈ beta_a / (beta_a + beta_b)``.  The destruction event fires
    when this fraction exceeds ``threshold``, so setting
    ``threshold ≈ beta_a / (beta_a + beta_b)`` gives P(destruction) ≈ 0.5.
    """
    rng = np.random.default_rng(seed)
    p_vec = rng.beta(beta_a, beta_b, size=n_cells).astype(np.float32)

    beta_mean = beta_a / (beta_a + beta_b)
    print(
        f"[synthetic_habitat] n_cells={n_cells:,}  "
        f"Beta({beta_a}, {beta_b})  mean≈{beta_mean:.4f}  threshold={threshold}"
    )

    habitat = CompactedHabitat(
        habitat_code=habitat_code,
        n_cells=n_cells,
        p_vec=p_vec,
    )
    return habitat, threshold


def make_overlapping_spatial_habitats(
    map_shape: tuple[int, int] = (200, 200),
    n_habitats: int = 80,
    block_frac: float = 0.22,
    beta_a: float = 2.0,
    beta_b: float = 5.0,
    threshold: float = 0.28,
    seed: int = 42,
) -> tuple[np.ndarray, list[SpatialHabitat]]:
    """Build a 2-D probability map and many *overlapping* spatial habitats.

    Each habitat is a randomly-placed rectangular block covering roughly
    ``block_frac`` of each map dimension.  With ``n_habitats`` such blocks on a
    modest map they overlap heavily, so the union of occupied cells is far
    smaller than the sum of per-habitat cell counts — exactly the regime where
    the Map-Centric kernel wins, because it draws one RNG sample per *map* cell
    instead of re-sampling every shared cell once per habitat.

    Parameters
    ----------
    map_shape : tuple[int, int]
        ``(rows, cols)`` of the geographic grid.
    n_habitats : int
        Number of habitats to generate.  Use ``> 64`` to exercise host batching.
    block_frac : float
        Side length of each habitat's rectangular block as a fraction of the
        map dimensions.  Larger ⇒ more overlap.
    beta_a, beta_b : float
        Shape parameters of the Beta distribution for ``p_map`` (see
        :func:`make_synthetic_habitat`).
    threshold : float
        Per-habitat destruction threshold θ, near the Beta mean for
        non-degenerate destruction probabilities.
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    p_map : numpy.ndarray
        2-D ``float32`` invasion-probability map of shape ``map_shape``.
    habitats : list[SpatialHabitat]
        ``n_habitats`` overlapping habitats aligned to ``p_map``.
    """
    rows, cols = map_shape
    rng = np.random.default_rng(seed)
    p_map = rng.beta(beta_a, beta_b, size=map_shape).astype(np.float32)

    bh = max(1, int(round(block_frac * rows)))
    bw = max(1, int(round(block_frac * cols)))

    habitats: list[SpatialHabitat] = []
    union = np.zeros(map_shape, dtype=bool)
    total_cells = 0
    for h in range(n_habitats):
        r0 = int(rng.integers(0, max(1, rows - bh + 1)))
        c0 = int(rng.integers(0, max(1, cols - bw + 1)))
        mask = np.zeros(map_shape, dtype=bool)
        mask[r0 : r0 + bh, c0 : c0 + bw] = True
        habitats.append(
            SpatialHabitat(
                habitat_code=f"SYN{h:03d}",
                presence_mask=mask,
                threshold=threshold,
            )
        )
        union |= mask
        total_cells += int(mask.sum())

    n_union = int(union.sum())
    overlap_ratio = total_cells / n_union if n_union else 0.0
    print(
        f"[overlapping_habitats] map={rows}x{cols} ({rows * cols:,} cells)  "
        f"n_habitats={n_habitats}  block={bh}x{bw}  "
        f"sum(cells)={total_cells:,}  union={n_union:,}  "
        f"overlap≈{overlap_ratio:.2f}x  threshold={threshold}"
    )
    return p_map, habitats
