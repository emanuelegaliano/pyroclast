"""LWS & GWS Parameter Sweep Standalone Benchmark.

This script executes a dense grid parameter tuning sweep over various 
Local Work Size (LWS) and Global Work Size (GWS) combinations using the selected 1-D Monte Carlo kernel.
Results are saved to 'benchmark/lws_gws_benchmark.csv' for Jupyter Notebook plotting.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
)
from pyroclast.domain.models import GridTopology, MonteCarloConfig
from pyroclast.services import run_preprocessing_batch


KERNELS = {
    "1D-standard": PyOpenCLMonteCarloAdapter,
    "1D-ping-pong": PyOpenCLMonteCarloPingPongAdapter,
    # Vectorized-RNG variants (factories carry the vec_width).
    "1D-vec2": lambda profiling=False: PyOpenCLMonteCarloVectorizedAdapter(profiling=profiling, vec_width=2),
    "1D-vec4": lambda profiling=False: PyOpenCLMonteCarloVectorizedAdapter(profiling=profiling, vec_width=4),
    "1D-vec8": lambda profiling=False: PyOpenCLMonteCarloVectorizedAdapter(profiling=profiling, vec_width=8),
}

KERNEL_SELECTED = "1D-standard"

def main() -> None:
    

    # 2. Setup environment and load map data
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip('"\'')
    cache_dir = Path(os.getenv("CACHE_DIR", str(Path(data_path) / "cache")).strip('"\''))
    cache_dir.mkdir(parents=True, exist_ok=True)
    invasion_map = os.getenv("INVASION_MAP", "").strip('"\'') or None

    print("Executing Preprocessing pipeline to obtain compacted habitat maps...")
    repo = FileMapRepository(data_path, invasion_map=invasion_map)
    preprocess_adapter = PyOpenCLAdapter()
    compacted = run_preprocessing_batch(
        repo=repo,
        compute=preprocess_adapter,
        criteria=HabitatCriteria(),
        cache_dir=cache_dir,
    )

    if not compacted:
        print("Error: No habitats found in data path. Benchmark aborted.")
        sys.exit(1)
        
    target_habitat = compacted[0]
    print(f"Selected Habitat for Benchmarks: '{target_habitat.habitat_code}' ({target_habitat.n_cells:,} active cells)")

    kernel_label = KERNEL_SELECTED
    adapter_cls = KERNELS.get(kernel_label)
    if adapter_cls is None:
        print(f"Error: Invalid kernel selection '{kernel_label}'. Available options: {list(KERNELS.keys())}")
        sys.exit(1)
    adapter = adapter_cls(profiling=True)

    device = adapter._ctx.devices[0]
    gpu_name = device.name.strip()
    print(f"Detected GPU for benchmarking: '{gpu_name}'")
    print(f"Running LWS & GWS Parameter Tuning Sweep on '{kernel_label}' Monte Carlo kernel...")

    # 4. Sweep Parameters
    MC_RUNS_BENCH = 1_000_000
    THRESHOLD = 0.005
    SEED = 42

    lws_options = [64, 128, 256, 512]
    # step of 512
    gws_options = list(range(512, 16384 + 1, 512))

    bench_results = []
    
    for gws in gws_options:
        for lws in lws_options:
            topology = GridTopology(gws=gws, lws=lws)
            # Instantiate new MonteCarloConfig to prevent dataclass FrozenInstanceError
            config = MonteCarloConfig(
                n_runs=MC_RUNS_BENCH,
                threshold=THRESHOLD,
                seed=SEED,
                topology=topology
            )

        
            adapter.reset_profile()               
            adapter.run(target_habitat, config)

            bench = adapter.benchmark()[0]
            mean_ms = bench.mean_ms
            throughput = MC_RUNS_BENCH / (mean_ms / 1000.0)

            bench_results.append({
                "LWS": lws,
                "GWS": gws,
                "Time (ms)": mean_ms,
                "Throughput (Sim/s)": throughput,
                "GPU": gpu_name,
                "Kernel": kernel_label
            })
            print(f"  Completed LWS={lws}, GWS={gws} | Mean Time: {mean_ms:.2f} ms", end="\r")

    print("\nBenchmark sweep completed. Processing data...")
    df_bench = pd.DataFrame(bench_results)

    # 5. Calculate Standardized Time relative performance (Deviation from Mean %) at each GWS
    gws_means = df_bench.groupby("GWS")["Time (ms)"].mean().to_dict()
    df_bench["Deviation from Mean (%)"] = df_bench.apply(
        lambda row: (row["Time (ms)"] / gws_means[row["GWS"]] - 1) * 100,
        axis=1
    )

    # 6. Save results to CSV file
    csv_path = Path(__file__).parent / "lws_gws_benchmark.csv"
    df_bench.to_csv(csv_path, index=False)
    print(f"Successfully exported benchmark results to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
