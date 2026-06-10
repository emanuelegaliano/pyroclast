"""Master script to run all benchmark sweeps.

Runs both `benchmark_sim_scaling.py` and `benchmark_habitat_scaling.py`
and outputs the final execution summary.
"""

from __future__ import annotations

import argparse
import time
import re
from enum import IntEnum
from pathlib import Path

import benchmark_sim_scaling
import benchmark_habitat_scaling
import benchmark_size_scaling
import benchmark_rng_vs_norng
import benchmark_lws_gws
import benchmark_compaction
import benchmark_2d_topology
import benchmark_map_centric

SAVE_FIGURES = False


class SweepTask(IntEnum):
    """Enumeration of all benchmark sweeps available in the suite."""
    SIMULATION_SCALING = 1
    HABITAT_SCALING = 2
    SIZE_SCALING = 3
    LWS_GWS_TUNING = 4
    RNG_VS_NORNG = 5
    STREAM_COMPACTION = 6
    TOPOLOGY_2D = 7
    MAP_CENTRIC = 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyroclast Benchmark Suite Runner")
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Sweep number to start from (1-8). Default is 1."
    )
    args = parser.parse_args()
    start_idx = args.start

    if start_idx < 1 or start_idx > 8:
        print(f"Error: --start must be between 1 and 8, got {start_idx}")
        return

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
    print(f"Starting execution from Sweep {start_idx}...\n")

    if start_idx <= SweepTask.SIMULATION_SCALING:
        print("\n--- Running Sweep 1: Simulation Scaling (500k to 1M runs) ---")
        benchmark_sim_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.HABITAT_SCALING:
        print("\n--- Running Sweep 2: Habitat Scaling (1 to N habitats) ---")
        benchmark_habitat_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.SIZE_SCALING:
        print("\n--- Running Sweep 3: Size Scaling (Factors 10 to 1) ---")
        benchmark_size_scaling.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.LWS_GWS_TUNING:
        print("\n--- Running Sweep 4: LWS & GWS Parameter Tuning ---")
        benchmark_lws_gws.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.RNG_VS_NORNG:
        print("\n--- Running Sweep 5: RNG Overhead Analysis ---")
        benchmark_rng_vs_norng.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.STREAM_COMPACTION:
        print("\n--- Running Sweep 6: Stream Compaction Variants ---")
        benchmark_compaction.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.TOPOLOGY_2D:
        print("\n--- Running Sweep 7: 2D Topology Analysis ---")
        benchmark_2d_topology.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    if start_idx <= SweepTask.MAP_CENTRIC:
        print("\n--- Running Sweep 8: Map-Centric vs Run-Centric Scaling ---")
        benchmark_map_centric.main(results_dir=results_dir, save_figures=SAVE_FIGURES)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"All requested benchmarks completed successfully in {elapsed:.1f} seconds.")
    print(f"Output files generated in `{results_dir}/`:")
    if start_idx <= SweepTask.SIMULATION_SCALING:
        print(f"  - {results_dir}/simulation_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.HABITAT_SCALING:
        print(f"  - {results_dir}/habitat_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.SIZE_SCALING:
        print(f"  - {results_dir}/size_scaling.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.LWS_GWS_TUNING:
        print(f"  - {results_dir}/lws_gws_sweep.csv" + (" / lws_gws_heatmap.png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.RNG_VS_NORNG:
        print(f"  - {results_dir}/rng_vs_norng.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.STREAM_COMPACTION:
        print(f"  - {results_dir}/compaction_benchmark.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.TOPOLOGY_2D:
        print(f"  - {results_dir}/benchmark_2d_topology.csv" + (" / .png" if SAVE_FIGURES else ""))
    if start_idx <= SweepTask.MAP_CENTRIC:
        print(f"  - {results_dir}/benchmark_map_centric.csv" + (" / .png" if SAVE_FIGURES else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
