"""LWS & GWS Parameter Sweep Benchmark.

Performs a dense parameter sweep over various Local Work Size (LWS) and Global Work Size (GWS)
combinations using OpenCL event profiling. Generates a performance heatmap.
Saves results to `csv_results/lws_gws_sweep.csv` and produces a plot saved to `csv_results/lws_gws_heatmap.png`.
"""

from __future__ import annotations

import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyroclast import (
    PyOpenCLAdapter,
    PyOpenCLMonteCarloCommutativeAdapter,
    generate_synthetic_habitat_dem,
    generate_synthetic_dem,
)
from pyroclast.domain.models import GridTopology, MonteCarloConfig


def main(results_dir: Path | str | None = None, save_figures: bool = True) -> None:
    load_dotenv()
    print("Generating fully synthetic DEM...")
    dem = generate_synthetic_dem(shape=(2000, 2000))

    # 2. Generate synthetic habitat
    print("Generating synthetic habitat based on DEM...")
    hab_map, inv_map = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=0.3,
        mean_p=0.5,
        seed=42,
        habitat_code="SYNTH_LWS_GWS",
    )

    preprocess_adapter = PyOpenCLAdapter()
    compacted = preprocess_adapter.batch_preprocess(inv_map, [hab_map])
    target_habitat = compacted[0]
    print(f"Target Habitat: '{target_habitat.habitat_code}' ({target_habitat.n_cells:,} active cells)")

    # 3. Setup configurations
    mc_runs = 1048576  # 2^20 (approx 1M) simulations for clean power-of-two scaling
    threshold = 0.005
    seed = 42

    adapter = PyOpenCLMonteCarloCommutativeAdapter(profiling=True)
    device = adapter._ctx.devices[0]
    gpu_name = device.name.strip()
    print(f"Hardware: {gpu_name} (Max Compute Units: {device.max_compute_units})")

    if results_dir is None:
        import re
        sanitized = re.sub(r'[^\w\-_.]', '_', gpu_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        gpu_folder = sanitized.strip('_')
        results_dir = Path("csv_results") / gpu_folder
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # 4. Define parameter options
    lws_options = [64, 128, 256, 512]
    # We sweep GWS to cover under-saturated to fully saturated GPU threads
    gws_options = [16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    print("Running LWS & GWS parameter sweep...")
    bench_results = []

    for gws in gws_options:
        for lws in lws_options:
            if gws < lws:
                continue

            topology = GridTopology(gws=gws, lws=lws)
            config = MonteCarloConfig(
                n_runs=mc_runs,
                threshold=threshold,
                seed=seed,
                topology=topology,
            )

            trial_times = []
            for _ in range(3):
                adapter.reset_profile()
                adapter.run(target_habitat, config)
                
                bench_results_gpu = adapter.benchmark()
                gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results_gpu)
                trial_times.append(gpu_time)
            
            best_time = min(trial_times)
            throughput = mc_runs / (best_time / 1000.0)

            bench_results.append({
                "LWS": lws,
                "GWS": gws,
                "Time (ms)": best_time,
                "Throughput (Sim/s)": throughput,
            })
            print(f"  Completed LWS={lws:<4} GWS={gws:<6} | GPU Time: {best_time:7.2f} ms")

    # 5. Process results
    df = pd.DataFrame(bench_results)
    
    # Save CSV
    csv_path = results_dir / "lws_gws_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 6. Pivot for Heatmap plotting
    if save_figures:
        pivot_df = df.pivot(index="GWS", columns="LWS", values="Throughput (Sim/s)")
        # Divide throughput by 1e6 to represent in Millions of simulations per second
        pivot_df = pivot_df / 1e6

        # 7. Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Throughput (Million Sim/s)"},
            linewidths=0.5,
        )
        plt.title(f"Parameter Tuning Heatmap (Throughput vs. LWS/GWS)\nGPU: {gpu_name}")
        plt.xlabel("Local Work Size (LWS)")
        plt.ylabel("Global Work Size (GWS)")

        plt.tight_layout()
        plot_path = results_dir / "lws_gws_heatmap.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved heatmap image to: {plot_path}")


if __name__ == "__main__":
    main()
