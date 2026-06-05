"""Master script to run all benchmark sweeps.

Runs both `benchmark_sim_scaling.py` and `benchmark_habitat_scaling.py`
and outputs the final execution summary.
"""

from __future__ import annotations

import time
import re
from pathlib import Path

from benchmark import (
    benchmark_sim_scaling,
    benchmark_habitat_scaling,
    benchmark_size_scaling,
    benchmark_rng_vs_norng,
    benchmark_lws_gws,
    benchmark_compaction,
)

SAVE_FIGURES = False


def main() -> None:
    print("=" * 60)
    print("        PYROCLAST BENCHMARK SUITE RUNNER")
    print("=" * 60)
    t0 = time.perf_counter()

    # Detect GPU/device name for output subfolder
    try:
        from pyroclast import PyOpenCLAdapter
        temp_adapter = PyOpenCLAdapter()
        gpu_name = temp_adapter._ctx.devices[0].name.strip()
    except Exception:
        gpu_name = "unknown_gpu"

    sanitized = re.sub(r'[^\w\-_.]', '_', gpu_name)
    sanitized = re.sub(r'_+', '_', sanitized)
    gpu_folder = sanitized.strip('_')
    results_dir = Path("csv_results") / gpu_folder
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected GPU: {gpu_name}")
    print(f"Results will be saved in: {results_dir}/")
    print(f"Save Figures: {SAVE_FIGURES}")

    # 1. Run Simulation Scaling Sweep
    print("\n--- Running Sweep 1: Simulation Scaling (500k to 1M runs) ---")
    benchmark_sim_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    # 2. Run Habitat Scaling Sweep
    print("\n--- Running Sweep 2: Habitat Scaling (1 to N habitats) ---")
    benchmark_habitat_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    # 3. Run Size Scaling Sweep
    print("\n--- Running Sweep 3: Size Scaling (Factors 10 to 1) ---")
    benchmark_size_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    # 4. Run LWS/GWS Sweep
    print("\n--- Running Sweep 4: LWS & GWS Parameter Tuning ---")
    benchmark_lws_gws.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    # 5. Run RNG vs No-RNG Sweep
    print("\n--- Running Sweep 5: RNG Overhead Analysis ---")
    benchmark_rng_vs_norng.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    # 6. Run Stream Compaction Preprocessing Sweep
    print("\n--- Running Sweep 6: Stream Compaction Variants ---")
    benchmark_compaction.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"All benchmarks completed successfully in {elapsed:.1f} seconds.")
    print(f"Output files generated in `{results_dir}/`:")
    print(f"  - {results_dir}/simulation_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    print(f"  - {results_dir}/habitat_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    print(f"  - {results_dir}/size_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    print(f"  - {results_dir}/lws_gws_sweep.csv" + (" / lws_gws_heatmap.png" if SAVE_FIGURES else ""))
    print(f"  - {results_dir}/rng_vs_norng.csv" + (" / .png" if SAVE_FIGURES else ""))
    print(f"  - {results_dir}/compaction_benchmark.csv" + (" / .png" if SAVE_FIGURES else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
