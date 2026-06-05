"""Size scaling benchmark.

Sweeps the habitat dimension (by slicing the DEM by factors of 16, 8, 4, 2, 1)
for six MC configurations: Standard, Ping-Pong, Commutative, Vec-w2, VecPP-w2, and Multi-Hab Comm.
Saves results to `csv_results/size_scaling.csv` and produces a plot saved to `csv_results/size_scaling.png`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from pyroclast import (
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloCommutativeAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
    generate_synthetic_habitat_dem,
)
from pyroclast.domain.models import GridTopology, MonteCarloConfig


def main(results_dir: Path | str | None = None, save_figures: bool = True) -> None:
    load_dotenv()
    if results_dir is None:
        import re
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            devices = []
            for p in platforms:
                devices.extend(p.get_devices(cl.device_type.GPU))
            if not devices:
                for p in platforms:
                    devices.extend(p.get_devices())
            if devices:
                gpu_name = devices[0].name.strip()
            else:
                gpu_name = "unknown_gpu"
        except Exception:
            gpu_name = "unknown_gpu"
        sanitized = re.sub(r'[^\w\-_.]', '_', gpu_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        gpu_folder = sanitized.strip('_')
        results_dir = Path("csv_results") / gpu_folder
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = os.getenv("DATA_PATH", "data").strip('"\'')
    dem_path = os.getenv("DEM_PATH")
    if not dem_path:
        dem_path = str(Path(data_path) / "dem.tif")

    if not Path(dem_path).is_file():
        raise FileNotFoundError(
            f"DEM file not found at: {dem_path}. Please check DEM_PATH in .env or data folder."
        )

    # 1. Load DEM
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    # 2. Instantiate all adapters with profiling=True
    adapters = {
        "Standard": PyOpenCLMonteCarloAdapter(profiling=True),
        "Ping-Pong": PyOpenCLMonteCarloPingPongAdapter(profiling=True),
        "Commutative": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),
        "Vec-w2": PyOpenCLMonteCarloVectorizedAdapter(profiling=True, vec_width=2),
        "VecPP-w2": PyOpenCLMonteCarloVectorizedPingPongAdapter(profiling=True, vec_width=2),
        "Multi-Hab Comm": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),
    }

    # 3. Sweep scaling factors (e.g. 10.0 to 1.0, reducing size step by step)
    # Higher factor means more aggressive downsampling (smaller habitat)
    scaling_factors = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
    
    mc_runs = 500000
    threshold = 0.005
    seed = 42

    print(f"Running size scaling sweep across factors: {scaling_factors}...")
    raw_results = []

    # Warm-up runs with the smallest habitat to compile kernels
    print("Performing warm-up runs...")
    warmup_map, warmup_inv = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.3, mean_p=0.5, seed=42, downscale_factor=10.0)
    preprocess_adapter = PyOpenCLAdapter()
    warmup_compacted = preprocess_adapter.batch_preprocess(warmup_inv, [warmup_map])[0]
    
    warmup_config = MonteCarloConfig(n_runs=10000, threshold=threshold, seed=seed)
    for name, adapter in adapters.items():
        if name == "Multi-Hab Comm":
            adapter.run_multi_habitats([warmup_compacted], warmup_config)
        else:
            adapter.run(warmup_compacted, warmup_config)

    # Main Sweep Loop
    for factor in scaling_factors:
        # Generate synthetic habitat downscaled by `factor`
        hab_map, inv_map = generate_synthetic_habitat_dem(
            dem=dem,
            occupancy_fraction=0.3,
            mean_p=0.5,
            seed=42,
            habitat_code=f"SZ_{factor}",
            downscale_factor=factor
        )

        compacted = preprocess_adapter.batch_preprocess(inv_map, [hab_map])[0]
        n_cells = compacted.n_cells
        print(f"  Sweeping downscale factor = {factor:<4} | Active cells = {n_cells:,}")

        times_at_size = {}

        # Set workgroups to a canonical target to equalize comparison
        device = next(iter(adapters.values()))._ctx.devices[0]
        n_wg_target = device.max_compute_units * 8
        canonical_topo = GridTopology(gws=n_wg_target * 256, lws=256)

        config = MonteCarloConfig(
            n_runs=mc_runs,
            threshold=threshold,
            seed=seed,
            topology=canonical_topo,
        )

        for name, adapter in adapters.items():
            trial_times = []
            for _ in range(3):
                adapter.reset_profile()
                if name == "Multi-Hab Comm":
                    adapter.run_multi_habitats([compacted], config)
                else:
                    adapter.run(compacted, config)
                bench_results = adapter.benchmark()
                gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
                trial_times.append(gpu_time)
            
            times_at_size[name] = min(trial_times)

        # Calculate mean across all adapters at this step
        mean_time = np.mean(list(times_at_size.values()))

        for name, elapsed in times_at_size.items():
            deviation = (elapsed / mean_time - 1.0) * 100.0
            raw_results.append({
                "Factor": factor,
                "Active_Cells": n_cells,
                "Kernel": name,
                "Time (ms)": elapsed,
                "Deviation (%)": deviation,
            })

    # 4. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "size_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 5. Plotting
    if save_figures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Distinct markers and colors for plotting (matching the style)
        style_opts = {
            "Standard": {"color": "tab:red", "marker": "v"},
            "Ping-Pong": {"color": "darkorange", "marker": "s"},
            "Commutative": {"color": "tab:blue", "marker": "D"},
            "Vec-w2": {"color": "tab:brown", "marker": "X"},
            "VecPP-w2": {"color": "tab:pink", "marker": "*"},
            "Multi-Hab Comm": {"color": "tab:green", "marker": "o"},
        }

        # Plot Subplot 1: Execution Time Scaling
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax1.plot(
                sub_df["Active_Cells"] / 1000.0,
                sub_df["Time (ms)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax1.set_title("Execution Time vs. Habitat Size")
        ax1.set_xlabel("Active Cells (thousands)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # Plot Subplot 2: Relative Performance (Standardized Time)
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax2.plot(
                sub_df["Active_Cells"] / 1000.0,
                sub_df["Deviation (%)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, label="Group Mean")
        ax2.set_title("Relative Performance (Standardized Time)")
        ax2.set_xlabel("Active Cells (thousands)")
        ax2.set_ylabel("Deviation from Mean (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plot_path = results_dir / "size_scaling.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")


if __name__ == "__main__":
    main()
