"""Statistical Robustness Benchmark — Z-test and Chi-Square Test.

This script tests the **statistical equivalence** between the scalar baseline
kernel (V1 Standard) and every vectorized variant (Vec-w2/4/8, VecPP-w2/4/8)
on a **single synthetic habitat** (see ``benchmark/synthetic_habitat.py``).

Using a synthetic habitat rather than real data ensures:

* **Reproducibility** — results are independent of the data files on disk.
* **Non-degeneracy** — the Beta-distributed p_vec guarantees 0 < p̂ < 1, which
  is required for both statistical tests to be well-defined.
* **Uniformity** — all benchmark scripts use the same habitat, making
  cross-script comparisons meaningful.

Pipeline
--------
1. Build a ``CompactedHabitat`` via :func:`~benchmark.synthetic_habitat.make_synthetic_habitat`.
2. Run *R* simulations with the scalar baseline and each vectorized variant.
3. Record raw destruction counts ``D_scalar`` / ``D_vec``.
4. Perform for every (scalar, vectorized) pair:

   a. **Two-proportion Z-test** — H₀: p̂_scalar = p̂_vec.
   b. **Chi-square test** — independence in the 2×2 contingency table.

5. Save all results to ``benchmark/statistical_robustness.csv``.

Usage
-----
Run from the project root::

    python benchmark/statistical_robustness.py

The output CSV is consumed by ``benchmark_analysis.ipynb`` for plotting.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

# Make sure the benchmark package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_habitat import make_synthetic_habitat  # noqa: E402

from pyroclast import (
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
)
from pyroclast.domain.models import GridTopology, MonteCarloConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_topology(adapter, n_wg_target: int) -> GridTopology:
    """Return a GridTopology with ``n_wg_target`` work-groups for the adapter."""
    sample = adapter.suggest_topology(1)
    if isinstance(sample.lws, int):
        return GridTopology(gws=n_wg_target * 256, lws=256)
    return GridTopology(gws=(n_wg_target * 32, 8), lws=(32, 8))


def _run_and_count(adapter, habitat, config: MonteCarloConfig) -> int:
    """Run one MC pass and return the raw destruction count (not the probability).

    We exploit the fact that ``prob = count / n_runs`` and recover the integer
    count by rounding  prob * n_runs.  This is exact because the kernel itself
    works in integer arithmetic before the final division on the host.
    """
    prob = adapter.run(habitat, config)
    return round(prob * config.n_runs)


def _ztest_two_proportions(
    d1: int, n1: int, d2: int, n2: int
) -> tuple[float, float]:
    """Two-sided two-proportion Z-test.

    Returns ``(nan, nan)`` when the pooled proportion is 0 or 1 (i.e. both
    kernels produced the same degenerate outcome for every simulation), because
    the standard error is zero and the test is undefined.

    Returns
    -------
    z_stat : float
    p_value : float
    """
    p1 = d1 / n1
    p2 = d2 / n2
    p_pool = (d1 + d2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0.0:
        # Degenerate: pooled p is 0 or 1 — test is undefined.
        # Return NaN so callers can detect and report this edge case.
        return math.nan, math.nan
    z = (p1 - p2) / denom
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return z, p_val


def _chi2_contingency(
    d1: int, n1: int, d2: int, n2: int
) -> tuple[float, float]:
    """Chi-square test on the 2×2 contingency table.

    Table layout::

                Destruction   No destruction
        Scalar      d1           n1 - d1
        Vector      d2           n2 - d2

    Returns ``(nan, nan)`` when any row or column marginal is zero (e.g. all
    simulations resulted in destruction for both kernels), because the expected
    frequencies cannot be computed and the test is undefined.

    Returns
    -------
    chi2 : float
    p_value : float
    """
    table = np.array([[d1, n1 - d1], [d2, n2 - d2]], dtype=np.int64)
    # Guard: chi-square is undefined when any marginal total is zero.
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    if np.any(row_sums == 0) or np.any(col_sums == 0):
        return math.nan, math.nan
    chi2, p_val, _dof, _expected = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p_val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    R     = int(os.getenv("STAT_RUNS", "1000000"))
    SEED  = int(os.getenv("MC_SEED",   "42"))
    ALPHA = float(os.getenv("STAT_ALPHA", "0.05"))

    # Synthetic habitat parameters (override via env if needed)
    N_CELLS  = int(os.getenv("SYNTH_N_CELLS",  "50000"))
    BETA_A   = float(os.getenv("SYNTH_BETA_A", "2.0"))
    BETA_B   = float(os.getenv("SYNTH_BETA_B", "5.0"))
    SYNTH_SEED = int(os.getenv("SYNTH_SEED",   "0"))

    print(f"\n{'=' * 60}")
    print(f"  Statistical Robustness Benchmark")
    print(f"{'=' * 60}")
    print(f"  R={R:,}  seed={SEED}  α={ALPHA}")

    # ── Build synthetic habitat ──────────────────────────────────
    print("\n[1/3] Building synthetic habitat …")
    habitat, THRESHOLD = make_synthetic_habitat(
        n_cells=N_CELLS,
        beta_a=BETA_A,
        beta_b=BETA_B,
        seed=SYNTH_SEED,
    )
    print(f"  habitat_code={habitat.habitat_code!r}  "
          f"n_cells={habitat.n_cells:,}  threshold={THRESHOLD}")

    # ── Build adapters ───────────────────────────────────────────
    scalar_adapter = PyOpenCLMonteCarloAdapter(profiling=False)
    max_cu = scalar_adapter._ctx.devices[0].max_compute_units
    n_wg_target = max_cu * 8
    print(f"  GPU: max_cu={max_cu}  n_wg_target={n_wg_target}")

    vec_widths = [2, 4, 8]
    vectorized_adapters: list[tuple[str, object]] = []
    for w in vec_widths:
        vectorized_adapters.append(
            (f"Vec-w{w}",   PyOpenCLMonteCarloVectorizedAdapter(profiling=False, vec_width=w))
        )
        vectorized_adapters.append(
            (f"VecPP-w{w}", PyOpenCLMonteCarloVectorizedPingPongAdapter(profiling=False, vec_width=w))
        )

    # ── Run tests ────────────────────────────────────────────────
    print("\n[2/3] Running simulations and statistical tests …")
    rows: list[dict] = []

    print(f"\n  Habitat: {habitat.habitat_code}  (n_cells={habitat.n_cells:,})")

    # --- Scalar baseline ---
    scalar_cfg = MonteCarloConfig(
        n_runs=R,
        threshold=THRESHOLD,
        seed=SEED,
        topology=_canonical_topology(scalar_adapter, n_wg_target),
    )
    D_scalar = _run_and_count(scalar_adapter, habitat, scalar_cfg)
    p_scalar = D_scalar / R
    print(f"    [Scalar Standard]  D={D_scalar:,}  p̂={p_scalar:.6f}")

    # --- Each vectorized variant ---
    for vec_name, vec_adapter in vectorized_adapters:
        vec_cfg = MonteCarloConfig(
            n_runs=R,
            threshold=THRESHOLD,
            seed=SEED,
            topology=_canonical_topology(vec_adapter, n_wg_target),
        )
        D_vec = _run_and_count(vec_adapter, habitat, vec_cfg)
        p_vec = D_vec / R

        z_stat, p_ztest = _ztest_two_proportions(D_scalar, R, D_vec, R)
        chi2,   p_chi2  = _chi2_contingency(D_scalar, R, D_vec, R)

        reject_z    = (not math.isnan(p_ztest)) and (p_ztest < ALPHA)
        reject_chi2 = (not math.isnan(p_chi2))  and (p_chi2  < ALPHA)

        def _fmt_pval(p: float, reject: bool) -> str:
            if math.isnan(p):
                return "  N/A (degenerate)"
            return f"{p:.4f}{'*' if reject else ' '}"

        print(
            f"    [{vec_name:12s}]  D={D_vec:,}  p̂={p_vec:.6f}  "
            f"|Δp̂|={abs(p_vec - p_scalar):.6f}  "
            f"Z={z_stat:+.4f}  p_Z={_fmt_pval(p_ztest, reject_z)}  "
            f"χ²={chi2:.4f}  p_χ²={_fmt_pval(p_chi2, reject_chi2)}"
        )

        rows.append({
            "habitat":          habitat.habitat_code,
            "n_cells":          habitat.n_cells,
            "R":                R,
            "alpha":            ALPHA,
            "kernel_scalar":    "Standard",
            "kernel_vec":       vec_name,
            "D_scalar":         D_scalar,
            "D_vec":            D_vec,
            "p_scalar":         p_scalar,
            "p_vec":            p_vec,
            "delta_p":          p_vec - p_scalar,
            "abs_delta_p":      abs(p_vec - p_scalar),
            "z_stat":           z_stat,
            "p_value_ztest":    p_ztest,
            "reject_H0_ztest":  reject_z,
            "chi2_stat":        chi2,
            "p_value_chi2":     p_chi2,
            "reject_H0_chi2":   reject_chi2,
        })

    # ── Save results ─────────────────────────────────────────────
    print("\n[3/3] Saving results …")
    out_path = Path(__file__).parent / "statistical_robustness.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {out_path}")

    # ── Summary ──────────────────────────────────────────────────
    n_rejected_z    = sum(1 for r in rows if r["reject_H0_ztest"])
    n_rejected_chi2 = sum(1 for r in rows if r["reject_H0_chi2"])
    print(f"\n  Summary (α={ALPHA}):")
    print(f"    Total comparisons         : {len(rows)}")
    print(f"    H₀ rejected (Z-test)      : {n_rejected_z} / {len(rows)}")
    print(f"    H₀ rejected (χ²-test)     : {n_rejected_chi2} / {len(rows)}")
    if n_rejected_z == 0 and n_rejected_chi2 == 0:
        print("  ✓ All vectorized kernels are statistically equivalent to the scalar baseline.")
    else:
        print("  ✗ Some kernels differ significantly — investigate seeding or batching offsets.")


if __name__ == "__main__":
    main()
