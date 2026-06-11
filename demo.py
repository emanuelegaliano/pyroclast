#!/usr/bin/env python3
"""
Pyroclast Library Demo
----------------------
A unified entry point for testing and benchmarking the various OpenCL 
Monte Carlo kernels and stream compaction algorithms in Pyroclast.
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv

# Domain and Data Models
from pyroclast.domain.models import MonteCarloConfig, SpatialHabitat, GridTopology

# I/O and Synthetics
from pyroclast import (
    FileMapRepository, 
    HabitatCriteria, 
    InvasionCriteria,
    generate_synthetic_dem, 
    generate_synthetic_habitat_dem
)

# Monte Carlo Adapters
from pyroclast.adapters import (
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarlo2DAdapter,
    PyOpenCLMonteCarlo2DTransposedAdapter,
    PyOpenCLMonteCarloCommutativeAdapter,
    PyOpenCLMonteCarloContiguousAdapter,
    PyOpenCLMonteCarloGlobalSeedAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
    PyOpenCLMapCentricAdapter,
)

# Compaction Adapters
from pyroclast.adapters import (
    PyOpenCLHostScalarCompactionAdapter,
    PyOpenCLHostNonzeroCompactionAdapter,
    PyOpenCLHostCompressCompactionAdapter,
    PyOpenCLGPUScalarCompactionAdapter,
    PyOpenCLGPUVectorizedCompactionAdapter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMPACTION_MAP = {
    "host_scalar": PyOpenCLHostScalarCompactionAdapter,
    "host_nonzero": PyOpenCLHostNonzeroCompactionAdapter,
    "host_compress": PyOpenCLHostCompressCompactionAdapter,
    "gpu_scalar": PyOpenCLGPUScalarCompactionAdapter,
    "gpu_vectorized": PyOpenCLGPUVectorizedCompactionAdapter,
}

KERNEL_MAP = {
    "standard": PyOpenCLMonteCarloAdapter,
    "2d": PyOpenCLMonteCarlo2DAdapter,
    "2d_transposed": PyOpenCLMonteCarlo2DTransposedAdapter,
    "commutative": PyOpenCLMonteCarloCommutativeAdapter,
    "contiguous": PyOpenCLMonteCarloContiguousAdapter,
    "global_seed": PyOpenCLMonteCarloGlobalSeedAdapter,
    "map_centric": PyOpenCLMapCentricAdapter,
    "pingpong": PyOpenCLMonteCarloPingPongAdapter,
    "vectorized": PyOpenCLMonteCarloVectorizedAdapter,
    "vectorized_pingpong": PyOpenCLMonteCarloVectorizedPingPongAdapter,
}

def main():
    parser = argparse.ArgumentParser(description="Pyroclast OpenCL Benchmarking Demo")
    parser.add_argument("--monte_carlo", type=str, default="standard", choices=list(KERNEL_MAP.keys()),
                        help="Select the Monte Carlo kernel to execute.")
    parser.add_argument("--compaction", type=str, default="host_nonzero", choices=list(COMPACTION_MAP.keys()),
                        help="Select the Stream Compaction algorithm.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetically generated random data instead of loading from .env dataset.")
    parser.add_argument("--runs", type=int, default=10000,
                        help="Number of Monte Carlo simulations to run.")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Critical threshold for habitat destruction.")
    parser.add_argument("--multi", action="store_true",
                        help="Process multiple habitats concurrently if available.")
    parser.add_argument("--gws", type=int, default=65536,
                        help="Global Work Size (GWS) for Monte Carlo kernels.")
    parser.add_argument("--lws", type=int, default=256,
                        help="Local Work Size (LWS) for Monte Carlo kernels.")
    
    args = parser.parse_args()
    
    if args.monte_carlo in ("2d", "2d_transposed"):
        logger.warning(f"Kernel '{args.monte_carlo}' requires a 2D topology. Ignoring --gws and --lws, falling back to auto-topology.")
        config = MonteCarloConfig(n_runs=args.runs, threshold=args.threshold, seed=42)
    else:
        topology = GridTopology(gws=args.gws, lws=args.lws)
        config = MonteCarloConfig(n_runs=args.runs, threshold=args.threshold, seed=42, topology=topology)
    
    logger.info("=== Pyroclast Execution Demo ===")
    logger.info(f"Monte Carlo Kernel : {args.monte_carlo}")
    logger.info(f"Stream Compaction  : {args.compaction}")
    logger.info(f"Data Mode          : {'Synthetic' if args.synthetic else 'Real FileSystem (.env)'}")
    logger.info(f"Runs               : {args.runs}")
    logger.info(f"Threshold          : {args.threshold}")
    logger.info(f"Global Work Size   : {args.gws}")
    logger.info(f"Local Work Size    : {args.lws}")
    logger.info(f"Multi-Habitat      : {args.multi}\n")

    # 1. Data Loading
    if args.synthetic:
        logger.info("Generating synthetic DEMs...")
        dem = generate_synthetic_dem(shape=(2000, 2000), max_elevation=3000.0)
        hab_map1, inv_map = generate_synthetic_habitat_dem(
            dem=dem, occupancy_fraction=0.01, mean_p=0.5, seed=10, habitat_code="SYNTH_1"
        )
        if args.multi:
            hab_map2, _ = generate_synthetic_habitat_dem(
                dem=dem, occupancy_fraction=0.015, mean_p=0.5, seed=20, habitat_code="SYNTH_2"
            )
            hab_maps = [hab_map1, hab_map2]
        else:
            hab_maps = [hab_map1]
    else:
        logger.info("Loading real maps via FileMapRepository...")
        load_dotenv()
        data_path = os.getenv("DATA_PATH", "data")
        try:
            repo = FileMapRepository(data_path)
            inv_map = repo.get(InvasionCriteria())
            all_habs = repo.matching(HabitatCriteria())
            if not all_habs:
                raise ValueError("No habitats found in the repository!")
            hab_maps = all_habs if args.multi else [all_habs[0]]
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return
            
    # 2. Edge Case: Map-Centric
    if args.monte_carlo == "map_centric":
        logger.warning("Map-Centric kernel selected: Skipping stream compaction as it operates on raw spatial masks.")
        spatial_habitats = []
        for h in hab_maps:
            spatial_habitats.append(SpatialHabitat(
                habitat_code=h.code,
                presence_mask=h.data,
                threshold=args.threshold
            ))
            
        adapter = KERNEL_MAP["map_centric"](profiling=True)
        logger.info("Warming up OpenCL Map-Centric Adapter...")
        adapter.run_map(inv_map.data, spatial_habitats, config)
        
        logger.info("Executing Map-Centric Kernel...")
        t0 = time.perf_counter()
        results = adapter.run_map(inv_map.data, spatial_habitats, config)
        t1 = time.perf_counter()
        
        logger.info(f"Finished in {(t1-t0)*1000:.3f} ms")
        for code, prob in results.items():
            logger.info(f"Habitat {code} P(destruction) = {prob:.6f}")
        return

    # 3. Stream Compaction
    logger.info("Starting Stream Compaction preprocessing...")
    compaction_adapter = COMPACTION_MAP[args.compaction](profiling=True)
    
    t0 = time.perf_counter()
    compacted_habitats = compaction_adapter.batch_preprocess(inv_map, hab_maps)
    t1 = time.perf_counter()
    logger.info(f"Stream Compaction finished in {(t1-t0)*1000:.3f} ms")
    
    # 4. Monte Carlo Execution
    logger.info(f"Instantiating {args.monte_carlo} Monte Carlo Adapter...")
    mc_adapter = KERNEL_MAP[args.monte_carlo](profiling=True)
    
    logger.info("Warming up OpenCL kernels...")
    try:
        if args.multi and len(compacted_habitats) > 1:
            mc_adapter.run_multi_habitats(compacted_habitats, config)
        else:
            mc_adapter.run(compacted_habitats[0], config)
            
        logger.info("Executing timed Monte Carlo Kernel...")
        t0 = time.perf_counter()
        
        if args.multi and len(compacted_habitats) > 1:
            probs = mc_adapter.run_multi_habitats(compacted_habitats, config)
            t1 = time.perf_counter()
            logger.info(f"Finished multi-habitat run in {(t1-t0)*1000:.3f} ms")
            for h, prob in zip(compacted_habitats, probs):
                logger.info(f"Habitat {h.habitat_code} P(destruction) = {prob:.6f}")
        else:
            prob = mc_adapter.run(compacted_habitats[0], config)
            t1 = time.perf_counter()
            logger.info(f"Finished single-habitat run in {(t1-t0)*1000:.3f} ms")
            logger.info(f"Habitat {compacted_habitats[0].habitat_code} P(destruction) = {prob:.6f}")
    except (FileNotFoundError, NotImplementedError) as e:
        logger.error(f"Kernel {args.monte_carlo} does not support multi-habitat execution: {e}")


if __name__ == "__main__":
    main()
