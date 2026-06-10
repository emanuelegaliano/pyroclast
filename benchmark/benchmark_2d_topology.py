"""2D Topology Benchmark.

Sweeps the aspect ratio of the 2D work-group (cell_lanes vs run_lanes)
while keeping the total local work size (LWS) fixed at 256.
Compares the standard 2D kernel and the transposed 2D kernel
against the 1D Standard baseline.

Saves results to `csv_results/benchmark_2d_topology.csv` and produces
a plot saved to `csv_results/benchmark_2d_topology.png`.
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
    PyOpenCLMonteCarlo2DAdapter,
    PyOpenCLMonteCarlo2DTransposedAdapter,
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

    print("Generating synthetic habitat based on DEM...")
    hab_map, inv_map = generate_synthetic_habitat_dem(
        dem=dem,
        occupancy_fraction=0.3,
        mean_p=0.5,
        seed=42,
    )

    print("Preprocessing synthetic map on GPU...")
    preprocess_adapter = PyOpenCLAdapter()
    compacted = preprocess_adapter.batch_preprocess(inv_map, [hab_map])[0]
    
    # 2. Benchmark parameters
    mc_runs = 1_000_000
    threshold = 0.005
    seed = 42

    # Aspect ratios to sweep: (cell_lanes, run_lanes) that multiply to 256
    aspect_ratios = [(256, 1), (128, 2), (64, 4), (32, 8), (16, 16), (8, 32), (4, 64)]

    print(f"Running 2D topology sweep for {mc_runs} runs...")
    raw_results = []
    
    # We will instantiate adapters dynamically in the loop to pass the cell_lanes and run_lanes
    
    # Baseline: Standard 1D
    print("  Running Standard 1D (Baseline)...")
    standard_adapter = PyOpenCLMonteCarloAdapter(profiling=True)
    device = standard_adapter._ctx.devices[0]
    n_wg_target = device.max_compute_units * 8
    
    # Warm-up baseline
    warmup_config = MonteCarloConfig(n_runs=10000, threshold=0.005, seed=42)
    standard_adapter.run(compacted, warmup_config)
    
    # Run baseline
    standard_topo = GridTopology(gws=n_wg_target * 256, lws=256)
    config_std = MonteCarloConfig(n_runs=mc_runs, threshold=threshold, seed=seed, topology=standard_topo)
    
    trial_times = []
    for _ in range(3):
        standard_adapter.reset_profile()
        standard_adapter.run(compacted, config_std)
        bench_results = standard_adapter.benchmark()
        gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
        trial_times.append(gpu_time)
    
    base_time = min(trial_times)
    base_throughput = mc_runs / (base_time / 1000.0)
    
    for (cL, rL) in aspect_ratios:
        # We append baseline for every row so the line spans the chart
        raw_results.append({
            "Aspect Ratio": f"{cL}x{rL}",
            "Kernel": "Standard 1D (Baseline)",
            "Time (ms)": base_time,
            "Throughput (Sim/s)": base_throughput,
        })
        
        # Instantiate 2D adapters
        print(f"  Sweeping ratio {cL}x{rL}...")
        try:
            adapter_2d = PyOpenCLMonteCarlo2DAdapter(profiling=True, cell_lanes=cL, run_lanes=rL)
            adapter_2dt = PyOpenCLMonteCarlo2DTransposedAdapter(profiling=True, cell_lanes=cL, run_lanes=rL)
        except Exception as e:
            print(f"    Skipping {cL}x{rL} due to initialization error: {e}")
            continue
            
        adapters = {
            "2D Interleaved": adapter_2d,
            "2D Transposed": adapter_2dt
        }
        
        # Warmup
        for name, ad in adapters.items():
            try:
                ad.run(compacted, warmup_config)
            except Exception as e:
                pass
                
        # We let the adapters suggest their own topology based on max_compute_units and cell/run lanes
        config_2d = MonteCarloConfig(n_runs=mc_runs, threshold=threshold, seed=seed)
        
        for name, adapter in adapters.items():
            trial_times = []
            try:
                for _ in range(3):
                    adapter.reset_profile()
                    adapter.run(compacted, config_2d)
                    bench_results = adapter.benchmark()
                    gpu_time = sum(b.mean_ms * b.n_runs for b in bench_results)
                    trial_times.append(gpu_time)
                
                min_time = min(trial_times)
                throughput = mc_runs / (min_time / 1000.0)
                
                raw_results.append({
                    "Aspect Ratio": f"{cL}x{rL}",
                    "Kernel": name,
                    "Time (ms)": min_time,
                    "Throughput (Sim/s)": throughput,
                })
            except Exception as e:
                print(f"    Error running {name} at {cL}x{rL}: {e}")

    # 3. Process results
    df = pd.DataFrame(raw_results)
    
    # Save CSV
    csv_path = results_dir / "benchmark_2d_topology.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # 4. Plotting
    if save_figures and not df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        style_opts = {
            "Standard 1D (Baseline)": {"color": "black", "marker": None, "linestyle": "--"},
            "2D Interleaved": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
            "2D Transposed": {"color": "tab:orange", "marker": "s", "linestyle": "-"},
        }

        # X-axis will be categorical based on the aspect ratios
        x_labels = [f"{cL}x{rL}" for cL, rL in aspect_ratios]
        
        for name in style_opts.keys():
            sub_df = df[df["Kernel"] == name]
            if sub_df.empty:
                continue
            
            # Align with x_labels order
            y_vals = []
            for xl in x_labels:
                match = sub_df[sub_df["Aspect Ratio"] == xl]
                if not match.empty:
                    y_vals.append(match.iloc[0]["Time (ms)"])
                else:
                    y_vals.append(np.nan)
                    
            ax1.plot(
                x_labels,
                y_vals,
                label=name,
                color=style_opts[name]["color"],
                marker=style_opts[name]["marker"],
                linestyle=style_opts[name]["linestyle"],
                linewidth=2,
            )

        ax1.set_title("2D Kernel Performance vs Aspect Ratio (LWS=256)")
        ax1.set_xlabel("Aspect Ratio (cell_lanes x run_lanes)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        plt.tight_layout()
        plot_path = results_dir / "benchmark_2d_topology.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot image to: {plot_path}")

if __name__ == "__main__":
    main()
