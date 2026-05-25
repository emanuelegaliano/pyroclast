"""Habitat coverage vs. the invasion-probability map.

Answers two questions about the *raw* rasters (before compaction):

1. How much of the invasion-probability map do the habitat presence maps
   cover?  The "invasion footprint" is the set of cells with ``p > 0``;
   we measure how much of it falls inside at least one habitat, and how
   much is left uncovered.
2. How many "empty" cells/zones are there?  Empty cells are those with
   ``p == 0`` (no invasion probability).  With SciPy available we also
   count the connected empty *zones* (contiguous ``p == 0`` regions).

Run
---
    python benchmark/habitat_coverage.py

Honours the same .env knobs as the rest of the project: ``DATA_PATH``
and (optionally) ``INVASION_MAP``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria

try:
    from scipy.ndimage import label as _cc_label  # connected components
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_SCIPY = False


def _pct(part: int, whole: int) -> str:
    """Format ``part / whole`` as a percentage string (safe when whole == 0)."""
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a"


def main() -> None:
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip("\"' ")
    invasion_map = os.getenv("INVASION_MAP", "").strip("\"' ") or None

    repo = FileMapRepository(data_path, invasion_map=invasion_map)
    invasion = repo.get(InvasionCriteria())
    habitats = repo.matching(HabitatCriteria())

    # Invasion footprint: cells that carry any probability mass.
    p = invasion.data.astype(np.float32)
    total = p.size
    risk = p > 0.0
    n_risk = int(np.count_nonzero(risk))
    n_empty = total - n_risk

    print("=" * 72)
    print("INVASION-PROBABILITY MAP")
    print("=" * 72)
    print(f"  grid            : {p.shape[0]} x {p.shape[1]}  =  {total:,} cells")
    print(f"  footprint (p>0) : {n_risk:>10,} cells  ({_pct(n_risk, total)} of grid)")
    print(f"  empty    (p==0) : {n_empty:>10,} cells  ({_pct(n_empty, total)} of grid)")

    # Per-habitat breakdown + accumulate the union mask.
    union = np.zeros(p.shape, dtype=bool)
    print("\n" + "=" * 72)
    print("PER-HABITAT OVERLAP WITH THE FOOTPRINT")
    print("=" * 72)
    header = (
        f"  {'code':>8} | {'cells':>9} | {'at-risk(p>0)':>12} | "
        f"{'self-cov':>8} | {'p==0 in hab':>11} | {'% footprint':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for h in habitats:
        mask = h.data > 0
        union |= mask
        n_cells = int(np.count_nonzero(mask))
        at_risk = int(np.count_nonzero(mask & risk))
        empty_in = n_cells - at_risk  # habitat present but no invasion risk
        print(
            f"  {h.code:>8} | {n_cells:>9,} | {at_risk:>12,} | "
            f"{_pct(at_risk, n_cells)} | {empty_in:>11,} | {_pct(at_risk, n_risk)}"
        )

    # Coverage of the footprint by the *union* of all habitats.
    n_union = int(np.count_nonzero(union))
    covered = int(np.count_nonzero(union & risk))   # p>0 inside some habitat
    uncovered = n_risk - covered                    # p>0 outside every habitat
    habitat_empty = n_union - covered               # habitat over p==0 cells

    print("\n" + "=" * 72)
    print("COVERAGE OF THE FOOTPRINT BY THE HABITAT UNION")
    print("=" * 72)
    print(f"  habitat union          : {n_union:>10,} cells  ({_pct(n_union, total)} of grid)")
    print(f"  footprint covered      : {covered:>10,} cells  ({_pct(covered, n_risk)} of p>0)")
    print(f"  footprint NOT covered  : {uncovered:>10,} cells  ({_pct(uncovered, n_risk)} of p>0)")
    print(f"  habitat over empty area: {habitat_empty:>10,} cells  ({_pct(habitat_empty, n_union)} of union, p==0)")

    # "Empty zones": connected components of the p==0 region.
    print("\n" + "=" * 72)
    print("EMPTY ZONES")
    print("=" * 72)
    if _HAS_SCIPY:
        empty = ~risk
        # 4-connectivity (orthogonal neighbours). Pass a 3x3 ones structure
        # for 8-connectivity instead.
        _, n_zones = _cc_label(empty)
        # For context, also report how many disjoint invasion clusters exist.
        _, n_clusters = _cc_label(risk)
        print(f"  empty cells (p==0)               : {n_empty:>10,}")
        print(f"  connected empty zones (4-conn)   : {n_zones:>10,}")
        print(f"  connected invasion clusters      : {n_clusters:>10,}  (context)")
    else:
        print(f"  empty cells (p==0)               : {n_empty:>10,}")
        print("  (install scipy to count connected empty zones)")


if __name__ == "__main__":
    main()
