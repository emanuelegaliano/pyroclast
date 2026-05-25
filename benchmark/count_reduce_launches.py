"""Count how many times ``reduce_sum_int`` is launched per MC run.

The MC pipeline closes its global reduction by recursively launching the
shared ``reduce_sum_int`` kernel until a single int remains. The number
of launches depends only on ``n_wg`` (the count of work-groups produced
by the sampling kernel) and the reducer's local work-size (256), so:

    n_wg == 1          -> 0 launches  (the while-loop in _reduce_partial
                                       exits immediately)
    2   <= n_wg <= 256 -> 1 launch
    257 <= n_wg <= 65k -> 2 launches
    n_wg >  65_536     -> 3 launches

This script sweeps each MC adapter variant across five topologies chosen
to hit each regime, and prints a table comparing the observed launch count
(read off ``adapter._reduce_launches`` after each run with
``profiling=True``) against the predicted count.

All variants inherit ``_reduce_partial`` from
``PyOpenCLMonteCarloAdapter``, so the table is expected to be uniform
across columns — which is precisely what we want to confirm.

Usage
-----
    python benchmark/count_reduce_launches.py

Exits with status 1 if any observed count disagrees with the prediction.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from pyroclast.adapters.opencl_mc_adapter import (
    _REDUCE_LWS,
    PyOpenCLMonteCarloAdapter,
)
from pyroclast.adapters.opencl_mc_pingpong_adapter import (
    PyOpenCLMonteCarloPingPongAdapter,
)
from pyroclast.domain.models import (
    CompactedHabitat,
    GridTopology,
    MonteCarloConfig,
)


# (column label, adapter class)
ADAPTERS: list[tuple[str, type]] = [
    ("1d_base", PyOpenCLMonteCarloAdapter),
    ("1d_pp", PyOpenCLMonteCarloPingPongAdapter),
]

# (row label, target n_wg, GridTopology)
TOPOLOGIES: list[tuple[str, int, GridTopology]] = [
    ("tiny",     1,      GridTopology(gws=256,        lws=256)),
    ("small",    64,     GridTopology(gws=16_384,     lws=256)),
    ("boundary", 256,    GridTopology(gws=65_536,     lws=256)),
    ("medium",   1_024,  GridTopology(gws=262_144,    lws=256)),
    ("large",    70_000, GridTopology(gws=17_920_000, lws=256)),
]


def predicted_launches(n_wg: int) -> int:
    """Return the number of ``reduce_sum_int`` launches needed for ``n_wg``.

    Each launch divides the live element count by at most ``_REDUCE_LWS``.
    The host-side loop ``while n_elems > 1`` skips entirely when n_wg==1.
    """
    if n_wg <= 1:
        return 0
    return math.ceil(math.log(n_wg) / math.log(_REDUCE_LWS))


def make_habitat() -> CompactedHabitat:
    p = np.full(8, 0.5, dtype=np.float32)
    return CompactedHabitat(habitat_code="synthetic", n_cells=8, p_vec=p)


def run_one(adapter, topology: GridTopology) -> int:
    adapter.reset_profile()
    cfg = MonteCarloConfig(
        n_runs=10_000, threshold=0.5, seed=42, topology=topology
    )
    adapter.run(make_habitat(), cfg)
    return len(adapter._reduce_launches)


def main() -> int:
    try:
        adapters = [(label, cls(profiling=True))
                    for (label, cls) in ADAPTERS]
    except Exception as exc:
        print(f"OpenCL device unavailable, skipping: {exc}")
        return 0

    header_cols = ["topology", "n_wg"] + [lab for (lab, _) in ADAPTERS] + [
        "predicted"
    ]
    widths = [10, 8] + [max(8, len(lab) + 2) for (lab, _) in ADAPTERS] + [10]

    print("== Reduce-launch sweep ==\n")
    header = "  ".join(col.ljust(w) for col, w in zip(header_cols, widths))
    print(header)
    print("-" * len(header))

    all_match = True
    for (row_label, n_wg, topo) in TOPOLOGIES:
        observed = []
        for (_lab, adapter) in adapters:
            observed.append(run_one(adapter, topo))
        predicted = predicted_launches(n_wg)
        cells = [row_label, str(n_wg)] + [str(o) for o in observed] + [
            str(predicted)
        ]
        line = "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        match = all(o == predicted for o in observed)
        marker = "" if match else "  <-- MISMATCH"
        if not match:
            all_match = False
        print(line + marker)

    print()
    if all_match:
        print("All variants match the predicted count for every "
              "topology.")
        return 0
    print("Mismatch detected — see rows marked above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
