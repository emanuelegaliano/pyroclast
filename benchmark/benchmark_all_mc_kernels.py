import os
from pathlib import Path

from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloCascadingAdapter,
    PyOpenCLMonteCarlo2DAdapter,
    PyOpenCLMonteCarlo2DPingPongAdapter,
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
        print(f"  memory (VRAM): {bench.memory_mb:.2f} MB")
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

    # ── V1 Standard ──────────────────────────────────────────────
    mc_std = PyOpenCLMonteCarloAdapter(profiling=True)
    bench_std = _run_mc(mc_std, "V1 Standard", compacted, mc_config, n_batches)

    # ── V2 Ping-Pong ─────────────────────────────────────────────
    mc_pp = PyOpenCLMonteCarloPingPongAdapter(profiling=True)
    bench_pp = _run_mc(mc_pp, "V2 Ping-Pong", compacted, mc_config, n_batches)

    # ── V3 Cascading ─────────────────────────────────────────────
    mc_cas = PyOpenCLMonteCarloCascadingAdapter(profiling=True)
    bench_cas = _run_mc(mc_cas, "V3 Cascading", compacted, mc_config, n_batches)

    # ── V4 2-D Grid-Stride ───────────────────────────────────────
    mc_2d = PyOpenCLMonteCarlo2DAdapter(profiling=True)
    bench_2d = _run_mc(mc_2d, "V4 2-D Grid-Stride", compacted, mc_config, n_batches)

    # ── V5 2-D Ping-Pong ─────────────────────────────────────────
    mc_2d_pp = PyOpenCLMonteCarlo2DPingPongAdapter(profiling=True)
    bench_2d_pp = _run_mc(mc_2d_pp, "V5 2-D Ping-Pong", compacted, mc_config, n_batches)

    # ── Benchmark comparison ─────────────────────────────────────
    section("Benchmark comparison")
    all_runs = [
        ("V1 Standard",     bench_std),
        ("V2 Ping-Pong",    bench_pp),
        ("V3 Cascading",    bench_cas),
        ("V4 2-D Stride",   bench_2d),
        ("V5 2-D Ping-Pong", bench_2d_pp),
    ]

    if all(b for _, b in all_runs):
        for label, bench in all_runs:
            print(f"\n  [ {label} ]")
            _print_bench(bench)

        def _total_ms(bench: list[BenchResult]) -> float:
            return sum(b.mean_ms for b in bench)

        total_std = _total_ms(bench_std)
        print("\n  Total mean ms (sampling + reduce):")
        for label, bench in all_runs:
            total = _total_ms(bench)
            speedup = total_std / total if total > 0 else float("nan")
            print(f"    {label:<16}: {total:7.3f} ms   ({speedup:.2f}x vs V1)")
    else:
        print("\n  Benchmark comparison skipped (not all kernels processed).")


if __name__ == "__main__":
    main()
