"""Stream Compaction Performance Benchmark.

Benchmarks different stream compaction implementations for preprocessing habitat masks:
- Host Python Scalar
- Host NumPy Baseline
- Host NumPy Nonzero
- Host NumPy Compress
- GPU Scalar Scan
- GPU Vectorized Scan
Saves results to `csv_results/compaction_benchmark.csv` and produces a plot saved to `csv_results/compaction_benchmark.png`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyroclast import (
    FileMapRepository,
    PyOpenCLAdapter,
    PyOpenCLHostScalarCompactionAdapter,
    PyOpenCLHostNonzeroCompactionAdapter,
    PyOpenCLHostCompressCompactionAdapter,
    PyOpenCLGPUScalarCompactionAdapter,
    PyOpenCLGPUVectorizedCompactionAdapter,
    InvasionCriteria,
    HabitatCriteria,
)


def main(results_dir: Path | str | None = None, save_figures: bool = True) -> None:
    load_dotenv()
    if results_dir is None:
        import re
        try:
            temp_adapter = PyOpenCLAdapter()
            device = temp_adapter._ctx.devices[0]
            gpu_name = device.name.strip()
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
    invasion_map = os.getenv("INVASION_MAP", "").strip('"\'') or None

    print(f"Loading repository maps from {data_path}...")
    repo = FileMapRepository(data_path, invasion_map=invasion_map)

    invasion = repo.get(InvasionCriteria())
    habitats = list(repo.matching(HabitatCriteria()))
    print(f"Loaded {len(habitats)} habitats for compaction benchmarking.")

    compaction_variants = {
        "Host NumPy Mask (Baseline)": PyOpenCLAdapter,
        "Host NumPy Nonzero": PyOpenCLHostNonzeroCompactionAdapter,
        "Host NumPy Compress": PyOpenCLHostCompressCompactionAdapter,
        "GPU Scalar Scan": PyOpenCLGPUScalarCompactionAdapter,
        "GPU Vectorized Scan": PyOpenCLGPUVectorizedCompactionAdapter,
    }

    compaction_rows = []
    N_WARMUP = 2
    N_BENCH = 10

    print("Running Stream Compaction Benchmark...")
    for label, cls in compaction_variants.items():
        adapter = cls(profiling=True)
        # Warmup
        for _ in range(N_WARMUP):
            adapter.batch_preprocess(invasion, habitats)
        # Benchmark
        adapter.reset_profile()
        for _ in range(N_BENCH):
            adapter.batch_preprocess(invasion, habitats)
            
        bench_results = adapter.benchmark()
        # Sum the mean times and memory usage
        mean_time = sum(b.mean_ms for b in bench_results)
        memory_mb = sum(b.memory_mb for b in bench_results)
        
        # Compute PCIe Device-to-Host transfer size in MB
        if "GPU" in label:
            # Multiplied by 4 because elements are float32/uint32 (4 Bytes)
            transfer_mb = sum(b.n_cells * 4 for b in bench_results) / 1e6
        else:
            transfer_mb = sum(invasion.data.size * 4 for b in bench_results) / 1e6
            
        compaction_rows.append({
            "Variant": label,
            "Time (ms)": mean_time,
            "Memory (VRAM MB)": memory_mb,
            "PCIe Transfer (MB)": transfer_mb,
        })
        print(f"  {label:<30}: {mean_time:.3f} ms")

    df_compaction = pd.DataFrame(compaction_rows)
    baseline_time = df_compaction[df_compaction["Variant"] == "Host NumPy Mask (Baseline)"]["Time (ms)"].values[0]
    df_compaction["Relative Speedup"] = baseline_time / df_compaction["Time (ms)"]

    # Save CSV
    csv_path = results_dir / "compaction_benchmark.csv"
    df_compaction.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # Plotting the results
    if save_figures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c", "#98df8a"]

        ax1.bar(df_compaction["Variant"], df_compaction["Time (ms)"], color=colors)
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_title("Absolute Preprocessing Time (Lower is Better)")
        ax1.set_xticklabels(df_compaction["Variant"], rotation=30, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        ax2.bar(df_compaction["Variant"], df_compaction["Relative Speedup"], color=colors)
        ax2.set_ylabel("Speedup vs Baseline")
        ax2.set_title("Relative Compaction Speedup (Higher is Better)")
        ax2.set_xticklabels(df_compaction["Variant"], rotation=30, ha="right")
        ax2.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="Baseline")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        plt.suptitle("Stream Compaction Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        plot_path = results_dir / "compaction_benchmark.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved compaction benchmark plot to: {plot_path}")


if __name__ == "__main__":
    main()
