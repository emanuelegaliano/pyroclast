"""RNG vs. No-RNG Benchmark.

Compares the GPU execution times of the Monte Carlo kernels with and without
random number generation (RNG) using OpenCL preprocessor macros (-DNO_RNG=1).
Isolates the computational overhead of RNG stream seeding and number generation.
Saves results to `csv_results/rng_vs_norng.csv` and produces a plot saved to `csv_results/rng_vs_norng.png`.
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

    # 2. Generate a synthetic habitat based on DEM
    print("Generating synthetic habitat based on DEM...")
    hab_map, inv_map = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=0.3,
        mean_p=0.5,
        seed=42,
        habitat_code="SYNTH_RNG",
    )

    preprocess_adapter = PyOpenCLAdapter()
    compacted = preprocess_adapter.batch_preprocess(inv_map, [hab_map])
    target_habitat = compacted[0]
    print(f"Target Habitat: '{target_habitat.habitat_code}' ({target_habitat.n_cells:,} active cells)")

    # 3. Setup configurations
    mc_runs = 500000
    threshold = 0.005
    seed = 42

    # Instantiate first to get max compute units for topology
    temp_adapter = PyOpenCLMonteCarloAdapter()
    device = temp_adapter._ctx.devices[0]
    gpu_name = device.name.strip()

    if results_dir is None:
        import re
        sanitized = re.sub(r'[^\w\-_.]', '_', gpu_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        gpu_folder = sanitized.strip('_')
        results_dir = Path("csv_results") / gpu_folder
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    n_wg_target = device.max_compute_units * 8
    canonical_topo = GridTopology(gws=n_wg_target * 256, lws=256)
    config = MonteCarloConfig(
        n_runs=mc_runs,
        threshold=threshold,
        seed=seed,
        topology=canonical_topo,
    )

    # 4. Instantiate adapters for both configurations
    # We test the primary kernel variants
    kernel_types = {
        "Standard": (PyOpenCLMonteCarloAdapter, {}),
        "Ping-Pong": (PyOpenCLMonteCarloPingPongAdapter, {}),
        "Commutative": (PyOpenCLMonteCarloCommutativeAdapter, {}),
        "Global-Seed": (PyOpenCLMonteCarloGlobalSeedAdapter, {}),
        "Vec-w2": (PyOpenCLMonteCarloVectorizedAdapter, {"vec_width": 2}),
        "Multi-Hab Comm": (PyOpenCLMonteCarloCommutativeAdapter, {}),
        "Multi-Hab GS": (PyOpenCLMonteCarloGlobalSeedAdapter, {}),
    }

    raw_results = []

    for name, (cls, kwargs) in kernel_types.items():
        print(f"Benchmarking kernel: {name}...")

        # A. Setup With RNG (Default)
        adapter_rng = cls(profiling=True, **kwargs)
        
        # B. Setup Without RNG (Compute Only)
        adapter_norng = cls(profiling=True, extra_build_options="-DNO_RNG=1", **kwargs)

        configurations = {
            "With RNG": adapter_rng,
            "Without RNG": adapter_norng,
        }

        for label, adapter in configurations.items():
            # Warmup
            if name in ("Multi-Hab Comm", "Multi-Hab GS"):
                adapter.run_multi_habitats([target_habitat], config)
            else:
                adapter.run(target_habitat, config)

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
            
            best_time = min(trial_times)
            raw_results.append({
                "Kernel": name,
                "Mode": label,
                "Time (ms)": best_time,
                "Throughput (Sim/s)": mc_runs / (best_time / 1000.0)
            })
            print(f"  {label:<12}: {best_time:.3f} ms")

    # 5. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "rng_vs_norng.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # Calculate percentages of RNG cost
    df_pivot = df.pivot(index="Kernel", columns="Mode", values="Time (ms)")
    df_pivot["RNG Overhead (ms)"] = df_pivot["With RNG"] - df_pivot["Without RNG"]
    df_pivot["RNG Cost (%)"] = (df_pivot["RNG Overhead (ms)"] / df_pivot["With RNG"]) * 100
    df_pivot["Speedup (x)"] = df_pivot["With RNG"] / df_pivot["Without RNG"]
    print("\n--- Summary of RNG Overhead ---")
    print(df_pivot[["With RNG", "Without RNG", "RNG Cost (%)", "Speedup (x)"]].to_string())

    # 6. Plotting
    if save_figures:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kernels = df_pivot.index
        x = np.arange(len(kernels))
        width = 0.35

        ax.bar(x - width/2, df_pivot["With RNG"], width, label="With RNG", color="tab:blue")
        ax.bar(x + width/2, df_pivot["Without RNG"], width, label="Without RNG (Compute Only)", color="tab:orange")

        # Add text labels for RNG cost on top of the bars
        for idx, (k, row) in enumerate(df_pivot.iterrows()):
            overhead_pct = row["RNG Cost (%)"]
            ax.text(
                idx,
                row["With RNG"] + (row["With RNG"] * 0.01),
                f"RNG: {overhead_pct:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                color="black"
            )

        ax.set_title("GPU Execution Time Comparison: With vs. Without RNG")
        ax.set_xlabel("Kernel Variant")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(kernels)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = results_dir / "rng_vs_norng.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")


if __name__ == "__main__":
    main()
