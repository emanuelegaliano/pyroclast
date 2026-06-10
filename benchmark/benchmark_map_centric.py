"""Map-Centric Benchmark.

Evaluates the Map-Centric architecture, which parallelizes across
spatial cells rather than simulation runs.

Sweeps the size of the habitat (n_cells) from small to extremely large
to find the cross-over threshold where Map-Centric outperforms the
Standard 1D run-centric baseline.

Saves results to `csv_results/benchmark_map_centric.csv` and produces
a plot saved to `csv_results/benchmark_map_centric.png`.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyroclast import (
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMapCentricAdapter,
    generate_synthetic_habitat_dem,
    generate_synthetic_dem,
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

    print("Generating fully synthetic DEM...")
    dem = generate_synthetic_dem(shape=(2000, 2000))

    print("Generating base synthetic habitat...")
    hab_map_base, inv_map_base = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=1.0, # dense so we have many cells
        mean_p=0.5,
        seed=42,
    )

    preprocess_adapter = PyOpenCLAdapter()
    base_compacted = preprocess_adapter.batch_preprocess(inv_map_base, [hab_map_base])[0]
    total_cells = base_compacted.n_cells

    # 2. Instantiate adapters
    adapters = {
        "Standard (Run-Centric)": PyOpenCLMonteCarloAdapter(profiling=True),
        "Map-Centric (Spatial)": PyOpenCLMapCentricAdapter(profiling=True),
    }

    # 3. Warm-up
    print("Performing warm-up runs...")
    warmup_config = MonteCarloConfig(n_runs=1000, threshold=0.005, seed=42)
    
    from pyroclast.domain.models import SpatialHabitat
    base_spatial_hab = SpatialHabitat(
        habitat_code=hab_map_base.code,
        presence_mask=hab_map_base.data,
        threshold=0.005
    )
    
    for name, adapter in adapters.items():
        if name == "Standard (Run-Centric)":
            adapter.run(base_compacted, warmup_config)
        else:
            adapter.run_map(inv_map_base.data, [base_spatial_hab], warmup_config)

    # 4. Sweep Parameters
    # Map-Centric scales with map size, so we sweep the map size by downscaling the compacted array
    # We fix n_runs to a moderate value because map-centric is expected to be slower on massive runs
    # but faster on massive map sizes.
    mc_runs = 50_000
    threshold = 0.005
    seed = 42

    # Scale factors: 1.0 (full), 0.5, 0.2, 0.1, 0.05, 0.01
    scale_factors = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"Running Map-Centric scaling sweep for {mc_runs} runs over map sizes...")
    raw_results = []
    
    device = adapters["Standard (Run-Centric)"]._ctx.devices[0]
    n_wg_target = device.max_compute_units * 8
    canonical_topo_std = GridTopology(gws=n_wg_target * 256, lws=256)
    
    # Run the sweep
    for scale in scale_factors:
        target_cells_1d = max(1, int(np.sqrt(total_cells * scale)))
        print(f"  Sweeping map dimension: {target_cells_1d}x{target_cells_1d} ({scale*100:.1f}%)")
        
        # Create a truncated 2D map for Map-Centric
        trunc_p_map = inv_map_base.data[:target_cells_1d, :target_cells_1d].astype(np.float32)
        # We also need to truncate the presence mask
        trunc_mask = hab_map_base.data[:target_cells_1d, :target_cells_1d]
        
        from pyroclast.domain.models import SpatialHabitat, CompactedHabitat
        trunc_spatial_hab = SpatialHabitat(
            habitat_code=f"TRUNC_{target_cells_1d}",
            presence_mask=trunc_mask,
            threshold=threshold
        )
        
        # Preprocess for the standard adapter
        # Need to create a fake RasterMap for batch_preprocess
        from pyroclast.services.synthetic import MemoryRasterMap
        trunc_inv_raster = MemoryRasterMap("inv", "invasion", trunc_p_map)
        trunc_hab_raster = MemoryRasterMap("hab", "habitat", trunc_mask)
        trunc_compacted = preprocess_adapter.batch_preprocess(trunc_inv_raster, [trunc_hab_raster])[0]
        actual_cells = trunc_compacted.n_cells
        
        if actual_cells == 0:
            print("    Skipping (0 active cells)")
            continue
            
        for name, adapter in adapters.items():
            if name == "Standard (Run-Centric)":
                config = MonteCarloConfig(n_runs=mc_runs, threshold=threshold, seed=seed, topology=canonical_topo_std)
                # Standard uses run() with compacted habitat
                trial_times = []
                for _ in range(3):
                    # We can't cleanly reset_profile for map centric since it doesn't log the same way, but standard does
                    adapter.reset_profile()
                    adapter.run(trunc_compacted, config)
                    bench_results = adapter.benchmark()
                    gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
                    trial_times.append(gpu_time)
            else:
                config = MonteCarloConfig(n_runs=mc_runs, threshold=threshold, seed=seed)
                # Map-Centric uses run_map() with spatial habitat
                import time
                trial_times = []
                for _ in range(3):
                    # It doesn't have a structured benchmark() output currently, we'll measure wall time
                    start_t = time.perf_counter()
                    adapter.run_map(trunc_p_map, [trunc_spatial_hab], config)
                    end_t = time.perf_counter()
                    trial_times.append((end_t - start_t) * 1000.0) # ms
            
            min_time = min(trial_times)
            throughput = mc_runs / (min_time / 1000.0)
            
            raw_results.append({
                "N_Cells": actual_cells,
                "Scale Fraction": scale,
                "Kernel": name,
                "Time (ms)": min_time,
                "Throughput (Sim/s)": throughput,
            })

    # 5. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "benchmark_map_centric.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 6. Plotting
    if save_figures and not df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        style_opts = {
            "Standard (Run-Centric)": {"color": "tab:red", "marker": "v", "linestyle": "-"},
            "Map-Centric (Spatial)": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
        }

        for name in style_opts.keys():
            sub_df = df[df["Kernel"] == name]
            if sub_df.empty:
                continue
            
            ax1.plot(
                sub_df["N_Cells"],
                sub_df["Time (ms)"],
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linestyle=style_opts[name]["linestyle"],
                linewidth=2,
            )

        ax1.set_title(f"Map-Centric vs Run-Centric Execution Time ({mc_runs} runs)")
        ax1.set_xlabel("Map Size (Number of Active Cells)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend()

        plt.tight_layout()
        plot_path = results_dir / "benchmark_map_centric.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")

if __name__ == "__main__":
    main()
