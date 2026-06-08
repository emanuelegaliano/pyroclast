"""Simulation scaling benchmark.

Sweeps the number of simulations (MC_RUNS) from 500,000 to 1,000,000 (step 50,000)
for five MC adapters: Standard, Ping-Pong, Commutative, Vec-w2, and VecPP-w2.
Saves the results to `csv_results/simulation_scaling.csv` and produces a plot
saved to `csv_results/simulation_scaling.png`.
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
    PyOpenCLMonteCarloGlobalSeedAdapter,
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

    # 1. Load DEM and generate synthetic habitat and invasion map
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    print("Generating synthetic habitat based on DEM...")
    hab_map, inv_map = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=0.3,
        mean_p=0.5,
        seed=42,
        habitat_code="SYNTH_DEM",
    )

    # Preprocess using the compute adapter
    print("Preprocessing synthetic maps on GPU...")
    preprocess_adapter = PyOpenCLAdapter()
    compacted = preprocess_adapter.batch_preprocess(inv_map, [hab_map])
    
    target_habitat = compacted[0]
    print(f"Generated Target Habitat: '{target_habitat.habitat_code}' ({target_habitat.n_cells:,} active cells)")

    # 2. Instantiate all adapters
    adapters = {
        "Standard": PyOpenCLMonteCarloAdapter(profiling=True),
        "Ping-Pong": PyOpenCLMonteCarloPingPongAdapter(profiling=True),
        "Commutative": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),
        "Global-Seed": PyOpenCLMonteCarloGlobalSeedAdapter(profiling=True),
        "Vec-w2": PyOpenCLMonteCarloVectorizedAdapter(profiling=True, vec_width=2),
        "VecPP-w2": PyOpenCLMonteCarloVectorizedPingPongAdapter(profiling=True, vec_width=2),
        "Multi-Hab Comm": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),
        "Multi-Hab GS": PyOpenCLMonteCarloGlobalSeedAdapter(profiling=True),
    }

    # 3. Warm-up runs to ensure GPU compilation & cache are hot
    print("Performing warm-up runs...")
    warmup_config = MonteCarloConfig(n_runs=10000, threshold=0.005, seed=42)
    for name, adapter in adapters.items():
        if name in ("Multi-Hab Comm", "Multi-Hab GS"):
            adapter.run_multi_habitats([target_habitat], warmup_config)
        else:
            adapter.run(target_habitat, warmup_config)


    # 4. Sweep Parameters
    mc_runs_sweep = list(range(500000, 1000000 + 1, 50000))
    threshold = 0.005
    seed = 42

    print(f"Running simulation scaling sweep from {mc_runs_sweep[0]:,} to {mc_runs_sweep[-1]:,}...")
    raw_results = []

    for mc_runs in mc_runs_sweep:
        print(f"  Sweeping MC_RUNS = {mc_runs:,}...")
        times_at_runs = {}

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
                if name in ("Multi-Hab Comm", "Multi-Hab GS"):
                    adapter.run_multi_habitats([target_habitat], config)
                else:
                    adapter.run(target_habitat, config)
                bench_results = adapter.benchmark()
                gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
                trial_times.append(gpu_time)
            
            times_at_runs[name] = min(trial_times)

        # Calculate mean across all adapters at this MC_RUNS step
        mean_time = np.mean(list(times_at_runs.values()))

        for name, elapsed in times_at_runs.items():
            deviation = (elapsed / mean_time - 1.0) * 100.0
            raw_results.append({
                "MC_RUNS": mc_runs,
                "Kernel": name,
                "Time (ms)": elapsed,
                "Deviation (%)": deviation,
            })

    # 5. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "simulation_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 6. Plotting
    if save_figures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Distinct markers and colors for plotting (matching the style of the user's graph)
        style_opts = {
            "Standard": {"color": "tab:red", "marker": "v"},
            "Ping-Pong": {"color": "darkorange", "marker": "s"},
            "Commutative": {"color": "tab:blue", "marker": "D"},
            "Global-Seed": {"color": "tab:purple", "marker": "^"},
            "Vec-w2": {"color": "tab:brown", "marker": "X"},
            "VecPP-w2": {"color": "tab:pink", "marker": "*"},
            "Multi-Hab Comm": {"color": "tab:green", "marker": "o"},
            "Multi-Hab GS": {"color": "teal", "marker": "p"},
        }


        # Plot Subplot 1: Execution Time Scaling
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax1.plot(
                sub_df["MC_RUNS"] / 1e6,
                sub_df["Time (ms)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax1.set_title("Execution Time Scaling")
        ax1.set_xlabel("Number of Simulations (MC_RUNS × 10⁶)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # Plot Subplot 2: Relative Performance (Standardized Time)
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax2.plot(
                sub_df["MC_RUNS"] / 1e6,
                sub_df["Deviation (%)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, label="Group Mean")
        ax2.set_title("Relative Performance (Standardized Time)")
        ax2.set_xlabel("Number of Simulations (MC_RUNS × 10⁶)")
        ax2.set_ylabel("Deviation from Mean (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plot_path = results_dir / "simulation_scaling.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")


if __name__ == "__main__":
    main()main()
