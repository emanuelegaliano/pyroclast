"""Habitat scaling benchmark.

Sweeps the number of habitats from 1 to N (all available habitats)
for six MC configurations:
  - Standard (iterative)
  - Ping-Pong (iterative)
  - Commutative (iterative)
  - Vec-w2 (iterative)
  - VecPP-w2 (iterative)
  - Multi-Hab Comm (parallel batch)

Saves results to `csv_results/habitat_scaling.csv` and produces a plot with
three subplots (absolute time, relative time, relative throughput) saved to
`csv_results/habitat_scaling.png`.
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

    # 1. Load DEM
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    # 2. Generate synthetic habitats and a single invasion map
    n_habitats_total = 10
    print(f"Generating {n_habitats_total} synthetic habitats based on DEM...")
    
    # Generate the invasion map first
    _, inv_map = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=0.0,
        mean_p=0.5,
        seed=42,
    )

    # Generate the different habitat presence masks
    hab_maps = []
    for i in range(n_habitats_total):
        # We vary seed and code, keeping occupancy_fraction at a realistic 20%
        hab_map, _ = generate_synthetic_habitat_dem(
            dem=dem,
            occupancy_fraction=0.2,
            mean_p=0.0,
            seed=42 + i + 1,
            habitat_code=f"SYNTH_DEM_{i}",
        )
        hab_maps.append(hab_map)

    # Preprocess all of them on the GPU
    print("Preprocessing synthetic maps on GPU...")
    preprocess_adapter = PyOpenCLAdapter()
    compacted = preprocess_adapter.batch_preprocess(inv_map, hab_maps)

    # 2. Instantiate all adapters
    adapters = {
        "Standard": PyOpenCLMonteCarloAdapter(profiling=True),
        "Ping-Pong": PyOpenCLMonteCarloPingPongAdapter(profiling=True),
        "Commutative": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),
        "Global-Seed": PyOpenCLMonteCarloGlobalSeedAdapter(profiling=True),
        "Vec-w2": PyOpenCLMonteCarloVectorizedAdapter(profiling=True, vec_width=2),
        "VecPP-w2": PyOpenCLMonteCarloVectorizedPingPongAdapter(profiling=True, vec_width=2),
        "Multi-Hab Comm": PyOpenCLMonteCarloCommutativeAdapter(profiling=True),  # Used for batch multi-run
        "Multi-Hab GS": PyOpenCLMonteCarloGlobalSeedAdapter(profiling=True),
    }

    # 3. Warm-up runs to ensure GPU compilation & cache are hot
    print("Performing warm-up runs...")
    warmup_config = MonteCarloConfig(n_runs=10000, threshold=0.005, seed=42)
    for name, adapter in adapters.items():
        if name in ("Multi-Hab Comm", "Multi-Hab GS"):
            adapter.run_multi_habitats(compacted[:1], warmup_config)
        else:
            adapter.run(compacted[0], warmup_config)


    # 4. Sweep Parameters
    # We sweep the number of active habitats from 1 up to N
    mc_runs = 500000
    threshold = 0.005
    seed = 42

    print(f"Running habitat scaling sweep from 1 to {n_habitats_total} habitats...")
    raw_results = []

    for n_habs in range(1, n_habitats_total + 1):
        print(f"  Sweeping N_HABITATS = {n_habs}...")
        active_compacted = compacted[:n_habs]
        times_at_habs = {}

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
                    adapter.run_multi_habitats(active_compacted, config)
                else:
                    # Run sequentially for the individual adapters
                    for hab in active_compacted:
                        adapter.run(hab, config)
                
                bench_results = adapter.benchmark()
                # Sum execution time of MC kernels
                gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
                trial_times.append(gpu_time)
            
            times_at_habs[name] = min(trial_times)

        # Calculate mean time and throughput across all adapters at this step
        mean_time = np.mean(list(times_at_habs.values()))
        
        # Throughput = (total simulations run) / time
        # Standard adapters run (mc_runs * n_habs) simulations sequentially
        # Multi-Hab Batch runs them concurrently but performs the same total simulations
        total_sims = mc_runs * n_habs
        throughputs = {name: total_sims / (t / 1000.0) for name, t in times_at_habs.items()}
        mean_throughput = np.mean(list(throughputs.values()))

        for name in adapters.keys():
            elapsed = times_at_habs[name]
            th = throughputs[name]
            deviation_time = (elapsed / mean_time - 1.0) * 100.0
            deviation_th = (th / mean_throughput - 1.0) * 100.0
            
            raw_results.append({
                "N_HABITATS": n_habs,
                "Kernel": name,
                "Time (ms)": elapsed,
                "Deviation Time (%)": deviation_time,
                "Throughput (Sim/s)": th,
                "Deviation Throughput (%)": deviation_th,
            })

    # 5. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "habitat_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 6. Plotting (3 subplots)
    if save_figures:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

        # Distinct markers and colors for plotting (matching styles)
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


        # Plot Subplot 1: Absolute Time
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax1.plot(
                sub_df["N_HABITATS"],
                sub_df["Time (ms)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax1.set_title("Execution Time Scaling")
        ax1.set_xlabel("Number of Habitats")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_xticks(range(1, n_habitats_total + 1))
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # Plot Subplot 2: Relative Performance (Standardized Time)
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax2.plot(
                sub_df["N_HABITATS"],
                sub_df["Deviation Time (%)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, label="Group Mean")
        ax2.set_title("Relative Time Performance")
        ax2.set_xlabel("Number of Habitats")
        ax2.set_ylabel("Deviation from Mean Time (%)")
        ax2.set_xticks(range(1, n_habitats_total + 1))
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        # Plot Subplot 3: Relative Throughput
        for name in adapters.keys():
            sub_df = df[df["Kernel"] == name]
            ax3.plot(
                sub_df["N_HABITATS"],
                sub_df["Deviation Throughput (%)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linewidth=1.5,
            )
        ax3.axhline(0, color="black", linestyle="-", linewidth=1.2, label="Group Mean")
        ax3.set_title("Relative Throughput Performance")
        ax3.set_xlabel("Number of Habitats")
        ax3.set_ylabel("Deviation from Mean Throughput (%)")
        ax3.set_xticks(range(1, n_habitats_total + 1))
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend()

        plt.tight_layout()
        plot_path = results_dir / "habitat_scaling.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")


if __name__ == "__main__":
    main()
