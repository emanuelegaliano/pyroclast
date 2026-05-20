import os
from pathlib import Path

from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    InvasionCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
)
from pyroclast.domain.models import BenchResult, MonteCarloConfig
from pyroclast.services import run_preprocessing_batch


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_bench(benches: list[BenchResult]) -> None:
    for bench in benches:
        print(f"  kernel      : {bench.kernel_name}")
        print(f"  launches    : {bench.n_runs}")
        print(f"  time (mean) : {bench.mean_ms:.3f} ms")
        print(f"  time (min)  : {bench.min_ms:.3f} ms")
        print(f"  bandwidth   : {bench.bandwidth_gbs:.2f} GB/s")
        print()


def _run_mc(adapter, label, compacted, config, n_batches) -> list[BenchResult]:
    section(f"Monte Carlo — {label}")
    if not compacted:
        print("  No habitats to process.")
        return []
    for habitat in compacted:
        def _progress(i, total, p, code=habitat.habitat_code):
            print(f"  [{code}]  {(i + 1) * 100 // total:3d}%  p≈{p:.4f}", end="\r", flush=True)
        prob = adapter.run_batched(habitat, config, n_batches, callback=_progress)
        print(f"  [{habitat.habitat_code}]  P(fraction > {config.threshold}) = {prob:.6f}    ")
    return adapter.benchmark()


def main() -> None:
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip('"\'')
    cache_dir = Path(os.getenv("CACHE_DIR", str(Path(data_path) / "cache")).strip('"\''))
    cache_dir.mkdir(parents=True, exist_ok=True)

    invasion_map = os.getenv("INVASION_MAP", "").strip('"\'') or None

    mc_config = MonteCarloConfig(
        n_runs=int(os.getenv("MC_RUNS", "1000000")),
        threshold=float(os.getenv("MC_THRESHOLD", "0.005")),
        seed=int(os.getenv("MC_SEED", "42")),
    )
    n_batches = int(os.getenv("MC_BATCHES", "10"))

    # ── Preprocessing (cached) ───────────────────────────────────
    section("Preprocessing")
    repo = FileMapRepository(data_path, invasion_map=invasion_map)
    preprocess_adapter = PyOpenCLAdapter()
    compacted = run_preprocessing_batch(
        repo=repo,
        compute=preprocess_adapter,
        criteria=HabitatCriteria(),
        cache_dir=cache_dir,
    )
    
    if not compacted:
        print("Error: No habitats found in data path.")
        return
    
    print(f"Habitats: {[h.habitat_code for h in compacted]}")
    print(f"Config  : R={mc_config.n_runs:,}  θ={mc_config.threshold}  seed={mc_config.seed}  batches={n_batches}")

    # ── 1-D kernel (Standard) ────────────────────────────────────
    mc_std = PyOpenCLMonteCarloAdapter(profiling=True)
    bench_std = _run_mc(mc_std, "Standard 1-D Kernel", compacted, mc_config, n_batches)

    # ── 1-D kernel (Ping-Pong) ───────────────────────────────────
    mc_pp = PyOpenCLMonteCarloPingPongAdapter(profiling=True)
    bench_pp = _run_mc(mc_pp, "Ping-Pong 1-D Kernel", compacted, mc_config, n_batches)

    # ── Benchmark comparison ─────────────────────────────────────
    section("Benchmark comparison")
    if bench_std and bench_pp:
        print("\n  [ Standard Kernel ]")
        _print_bench(bench_std)
        print("\n  [ Ping-Pong Kernel ]")
        _print_bench(bench_pp)

        mean_std = bench_std[0].mean_ms
        mean_pp = bench_pp[0].mean_ms
        print(f"\n  Speedup Ping-Pong vs Standard : {mean_std / mean_pp:.2f}x")
    else:
        print("\n  Benchmark comparison skipped (no habitats processed).")


if __name__ == "__main__":
    main()
