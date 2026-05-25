import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    InvasionCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
)
from pyroclast.domain.models import MonteCarloConfig
from pyroclast.services import run_preprocessing_batch
from pyroclast.services.monte_carlo import run_monte_carlo_batch

_MC_ADAPTER_MAP = {
    "standard": PyOpenCLMonteCarloAdapter,
    "ping_pong": PyOpenCLMonteCarloPingPongAdapter,
}


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip()
    force_recompute = os.getenv("FORCE_RECOMPUTE", "0").strip().lower() in ("1", "true", "yes")
    cache_dir = Path(os.getenv("CACHE_DIR", str(Path(data_path) / "cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Repository ────────────────────────────────────────────
    section("1. Repository — raw raster data")
    repo = FileMapRepository(data_path, invasion_map=os.getenv("INVASION_MAP"))

    invasion = repo.get(InvasionCriteria())
    p = invasion.data
    print(f"Invasion map  : shape={p.shape}, dtype={p.dtype}")
    print(f"  range       : [{p.min():.4f}, {p.max():.4f}]")
    print(f"  active cells: {np.count_nonzero(p):,}")

    habitats = repo.matching(HabitatCriteria())
    print(f"\nHabitats found: {[h.code for h in habitats]}")
    for h_map in habitats:
        h = h_map.data
        print(
            f"  [{h_map.code}]  presence={int(np.sum(h)):,} cells"
            f"  |  at-risk={int(np.sum((p > 0) & (h > 0))):,} cells"
        )

    # ── 2. GPU adapter (direct) ──────────────────────────────────
    section("2. PyOpenCLAdapter — direct batch_preprocess call")
    adapter = PyOpenCLAdapter(profiling=True)
    print(f"Device: {adapter._ctx.devices[0].name}")

    results = adapter.batch_preprocess(invasion, habitats)
    for r in results:
        print(
            f"  [{r.habitat_code}]  active={r.n_cells:,} cells"
            f"  |  Σp={r.total_probability:.2f}"
            f"  |  mean_p={r.mean_probability:.6f}"
        )

    # ── 3. Service layer with caching ────────────────────────────
    section("3. Service layer — first run (computes + writes cache)")
    compacted = run_preprocessing_batch(
        repo=repo,
        compute=adapter,
        criteria=HabitatCriteria(),
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    for r in compacted:
        cache_file = cache_dir / f"habitat_{r.habitat_code}.npy"
        print(
            f"  [{r.habitat_code}]  n_cells={r.n_cells:,}"
            f"  |  cached → {cache_file.name}"
            f"  ({cache_file.stat().st_size / 1024:.1f} KiB)"
        )

    section("4. Service layer — second run (all from cache, no GPU)")
    compacted_cached = run_preprocessing_batch(
        repo=repo,
        compute=adapter,
        criteria=HabitatCriteria(),
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    for r in compacted_cached:
        label = "forced recompute" if force_recompute else "from cache"
        print(f"  [{r.habitat_code}]  n_cells={r.n_cells:,}  ✓ {label}")

    # ── 5. CompactedHabitat Value Object properties ──────────────
    section("5. CompactedHabitat — Value Object properties")
    for r in compacted:
        print(f"  {r!r}")
        print(f"    total_probability : {r.total_probability:.4f}")
        print(f"    mean_probability  : {r.mean_probability:.6f}")
        print(f"    p_vec dtype/shape : {r.p_vec.dtype} {r.p_vec.shape}")
        above_mean = int(np.sum(r.p_vec > r.mean_probability))
        print(f"    cells above mean  : {above_mean:,}")

    # ── 6. Monte Carlo simulation ────────────────────────────────
    section("6. Monte Carlo — destruction probability per habitat")
    adapter_name = os.getenv("MC_ADAPTER", "standard").strip().lower()
    adapter_cls = _MC_ADAPTER_MAP.get(adapter_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown MC_ADAPTER value '{adapter_name}'. "
            f"Choose from: {list(_MC_ADAPTER_MAP.keys())}"
        )
    print(f"Using adapter: {adapter_cls.__name__}")
    mc_adapter = adapter_cls(profiling=True)
    mc_config = MonteCarloConfig(
        n_runs=int(os.getenv("MC_RUNS", "1000000000")),
        threshold=float(os.getenv("MC_THRESHOLD", "0.005")),
        seed=int(os.getenv("MC_SEED", "42")),
    )
    n_batches = int(os.getenv("MC_BATCHES", "10"))
    print(f"Config: R={mc_config.n_runs:,}  θ={mc_config.threshold}  seed={mc_config.seed}  batches={n_batches}")

    for habitat in compacted:
        def _progress(i, total, p, code=habitat.habitat_code):
            print(f"  [{code}]  {(i + 1) * 100 // total:3d}%  p≈{p:.4f}", end="\r", flush=True)
        try:
            prob = mc_adapter.run_batched(habitat, mc_config, n_batches, callback=_progress)
            print(f"  [{habitat.habitat_code}]  P(invaded_fraction > {mc_config.threshold}) = {prob:.4f}    ")
        except ValueError as exc:
            print(f"  [{habitat.habitat_code}]  SKIPPED: {exc}")

    # ── 7. Kernel benchmark ──────────────────────────────────────
    section("7. Kernel benchmark")
    preprocess_bench = adapter.benchmark()
    for pb in (preprocess_bench if isinstance(preprocess_bench, list) else [preprocess_bench]):
        print(f"  kernel      : {pb.kernel_name}")
        print(f"  shape       : {pb.shape[0]} x {pb.shape[1]}  ({pb.n_cells:,} cells)")
        print(f"  launches    : {pb.n_runs}")
        print(f"  time (mean) : {pb.mean_ms:.3f} ms")
        print(f"  time (min)  : {pb.min_ms:.3f} ms")
        print(f"  bandwidth   : {pb.bandwidth_gbs:.2f} GB/s", end="\n\n")

    try:
        for mc_bench in mc_adapter.benchmark():
            print(f"  kernel      : {mc_bench.kernel_name}")
            print(f"  shape       : {mc_bench.shape[0]} x {mc_bench.shape[1]}  ({mc_bench.n_cells:,} cells)")
            print(f"  launches    : {mc_bench.n_runs}")
            print(f"  time (mean) : {mc_bench.mean_ms:.3f} ms")
            print(f"  time (min)  : {mc_bench.min_ms:.3f} ms")
            print(f"  bandwidth   : {mc_bench.bandwidth_gbs:.2f} GB/s")
            print()
    except ValueError as exc:
        print(f"  [Monte Carlo Benchmark] SKIPPED: {exc}")


if __name__ == "__main__":
    main()
