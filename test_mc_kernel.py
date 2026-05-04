import os
from pathlib import Path

from dotenv import load_dotenv

from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria, PyOpenCLAdapter
from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.adapters.opencl_mc_2d_adapter import PyOpenCLMonteCarloAdapter2D
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


def _run_mc(adapter, label, compacted, config, n_batches) -> BenchResult:
    section(f"Monte Carlo — {label}")
    for habitat in compacted:
        def _progress(i, total, p, code=habitat.habitat_code):
            print(f"  [{code}]  {(i + 1) * 100 // total:3d}%  p≈{p:.4f}", end="\r", flush=True)
        prob = adapter.run_batched(habitat, config, n_batches, callback=_progress)
        print(f"  [{habitat.habitat_code}]  P(fraction > {config.threshold}) = {prob:.6f}    ")
    return adapter.benchmark()


def main() -> None:
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip('/"\'')
    cache_dir = Path(os.getenv("CACHE_DIR", str(Path(data_path) / "cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    mc_config = MonteCarloConfig(
        n_runs=int(os.getenv("MC_RUNS", "1000000")),
        threshold=float(os.getenv("MC_THRESHOLD", "0.005")),
        seed=int(os.getenv("MC_SEED", "42")),
    )
    n_batches = int(os.getenv("MC_BATCHES", "10"))

    if mc_config.n_runs > 1000:
        print(f"WARNING: This file is used just for benchmarking and may take a long time to run with the current configuration. {mc_config.n_runs} runs may take several minutes (and take most of your VRAM) or more, depending on the GPU. Consider reducing MC_RUNS for faster execution.")
        choice = input("Do you want to continue? [y/N] ")
        if choice.lower() not in ['y', 'yes']:
            exit(0)

    # ── Preprocessing (cached) ───────────────────────────────────
    section("Preprocessing")
    repo = FileMapRepository(data_path, invasion_map=os.getenv("INVASION_MAP"))
    invasion = repo.get(InvasionCriteria())
    habitats = repo.matching(HabitatCriteria())
    preprocess_adapter = PyOpenCLAdapter()
    compacted = run_preprocessing_batch(
        repo=repo,
        compute=preprocess_adapter,
        criteria=HabitatCriteria(),
        cache_dir=cache_dir,
    )
    print(f"Habitats: {[h.habitat_code for h in compacted]}")
    print(f"Config  : R={mc_config.n_runs:,}  θ={mc_config.threshold}  seed={mc_config.seed}  batches={n_batches}")

    # ── 1-D kernel ───────────────────────────────────────────────
    mc_1d = PyOpenCLMonteCarloAdapter(profiling=True)
    bench_1d = _run_mc(mc_1d, "1-D kernel (one work-item per run)", compacted, mc_config, n_batches)

    # ── 2-D kernel ───────────────────────────────────────────────
    mc_2d = PyOpenCLMonteCarloAdapter2D(profiling=True)
    bench_2d = _run_mc(mc_2d, "2-D kernel (one work-item per draw)", compacted, mc_config, n_batches)

    # ── Benchmark comparison ─────────────────────────────────────
    section("Benchmark comparison")
    print("\n  [ 1-D kernel ]")
    _print_bench(bench_1d)
    print("\n  [ 2-D kernel ]")
    _print_bench(bench_2d)

    mean_1d = bench_1d[0].mean_ms
    mean_2d = sum(b.mean_ms for b in bench_2d)
    print(f"\n  Speedup 2-D vs 1-D : {mean_1d / mean_2d:.2f}x  (total mean kernel time)")


if __name__ == "__main__":
    main()
