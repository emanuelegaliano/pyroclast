import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from benchmark.synthetic_habitat import make_overlapping_spatial_habitats
from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    PyOpenCLAdapter,
    PyOpenCLMapCentricAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
)
from pyroclast.domain.models import (
    BenchResult,
    CompactedHabitat,
    GridTopology,
    MonteCarloConfig,
)
from pyroclast.services import run_preprocessing_batch


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _canonical_topology(adapter, n_wg_target: int) -> GridTopology:
    """Build a GridTopology with `n_wg_target` work-groups for any MC adapter.

    Equalises the launch parallelism across the 1-D variants so that
    cross-kernel comparisons aren't confounded by adapter-specific defaults.
    wg_size stays 256, which the 1-D kernel requires via reqd_work_group_size.
    """
    return GridTopology(gws=n_wg_target * 256, lws=256)


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
    print(f"  topology: gws={config.topology.gws}  lws={config.topology.lws}")
    for habitat in compacted:
        def _progress(i, total, p, code=habitat.habitat_code):
            print(f"  [{code}]  {(i + 1) * 100 // total:3d}%  p≈{p:.4f}", end="\r", flush=True)
        try:
            prob = adapter.run_batched(habitat, config, n_batches, callback=_progress)
            print(f"  [{habitat.habitat_code}]  P(fraction > {config.threshold}) = {prob:.6f}    ")
        except ValueError as exc:
            print(f"  [{habitat.habitat_code}]  SKIPPED: {exc}")
    return adapter.benchmark()


def _with_topology(base: MonteCarloConfig, topology: GridTopology) -> MonteCarloConfig:
    """Return a new MonteCarloConfig sharing all fields of `base` except topology."""
    return MonteCarloConfig(
        n_runs=base.n_runs,
        threshold=base.threshold,
        seed=base.seed,
        topology=topology,
    )


def _run_map_centric_demo(n_runs: int, seed: int, n_habitats: int) -> None:
    """Compare Map-Centric (one batched sweep) vs. iterative Habitat-Centric.

    Uses a synthetic, heavily-overlapping multi-habitat scenario — the regime
    the Map-Centric kernel targets. Real preprocessing compacts away spatial
    indices, so we synthesise overlapping 2-D habitats here. End-to-end host
    wall-clock is timed (kernel launches + reduction) for a fair comparison.
    """
    section("Monte Carlo — V8 Map-Centric vs. iterative Habitat-Centric")

    p_map, habitats = make_overlapping_spatial_habitats(
        n_habitats=n_habitats, seed=seed
    )
    threshold = habitats[0].threshold
    cfg = MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed)
    n_chunks = -(-len(habitats) // 64)  # ceil division

    # ── Map-Centric: a single run_map call (internally chunked by 64) ──
    mc = PyOpenCLMapCentricAdapter()
    mc._queue.finish()
    t0 = time.perf_counter()
    probs_map = mc.run_map(p_map, habitats, cfg)
    mc._queue.finish()
    t_map = (time.perf_counter() - t0) * 1e3

    # ── Habitat-Centric: loop the standard kernel over compacted habitats ──
    std = PyOpenCLMonteCarloAdapter()
    p_flat = p_map.ravel()
    compacted: list[CompactedHabitat] = []
    for hab in habitats:
        idx = np.flatnonzero(hab.presence_mask.ravel())
        compacted.append(
            CompactedHabitat(
                habitat_code=hab.habitat_code,
                n_cells=int(idx.size),
                p_vec=np.ascontiguousarray(p_flat[idx], dtype=np.float32),
            )
        )
    std._queue.finish()
    t0 = time.perf_counter()
    probs_hc = {ch.habitat_code: std.run(ch, cfg) for ch in compacted}
    std._queue.finish()
    t_hc = (time.perf_counter() - t0) * 1e3

    max_diff = max(
        abs(probs_map[code] - probs_hc[code]) for code in probs_hc
    )
    speedup = t_hc / t_map if t_map > 0 else float("nan")

    print(f"  config           : R={n_runs:,}  θ={threshold}  seed={seed}")
    print(f"  habitats         : {len(habitats)}")
    print(f"  Habitat-Centric  : {t_hc:8.2f} ms  ({len(habitats)} kernel launches)")
    print(f"  Map-Centric      : {t_map:8.2f} ms  ({n_chunks} kernel launch(es))")
    print(f"  speedup          : {speedup:.2f}x vs Habitat-Centric")
    print(f"  max |Δp|         : {max_diff:.4f}  (statistical, streams differ)")


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
    max_cu = mc_std._ctx.devices[0].max_compute_units
    n_wg_target = max_cu * 8
    print(f"Canonical n_wg = {n_wg_target}  (max_cu={max_cu} × 8)")

    bench_std = _run_mc(
        mc_std, "V1 Standard", compacted,
        _with_topology(mc_config, _canonical_topology(mc_std, n_wg_target)),
        n_batches,
    )

    # ── V2 Ping-Pong ─────────────────────────────────────────────
    mc_pp = PyOpenCLMonteCarloPingPongAdapter(profiling=True)
    bench_pp = _run_mc(
        mc_pp, "V2 Ping-Pong", compacted,
        _with_topology(mc_config, _canonical_topology(mc_pp, n_wg_target)),
        n_batches,
    )

    # ── V7 Vectorized 1-D (vector RNG) ───────────────────────────────────
    bench_vec: dict[int, list[BenchResult]] = {}
    for w in (2, 4, 8):
        mc_vec = PyOpenCLMonteCarloVectorizedAdapter(profiling=True, vec_width=w)
        bench_vec[w] = _run_mc(
            mc_vec, f"V7 Vectorized (w={w})", compacted,
            _with_topology(mc_config, _canonical_topology(mc_vec, n_wg_target)),
            n_batches,
        )

    # ── Benchmark comparison ─────────────────────────────────────
    section("Benchmark comparison")
    all_runs = [
        ("V1 Standard",     bench_std),
        ("V2 Ping-Pong",    bench_pp),
        ("V7 Vec (w=2)",    bench_vec[2]),
        ("V7 Vec (w=4)",    bench_vec[4]),
        ("V7 Vec (w=8)",    bench_vec[8]),
    ]

    for label, bench in all_runs:
        if bench:
            print(f"\n  [ {label} ]")
            _print_bench(bench)
        else:
            print(f"\n  [ {label} ] - SKIPPED or NO DATA")

    def _total_ms(bench: list[BenchResult]) -> float:
        return sum(b.mean_ms for b in bench)

    total_std = _total_ms(bench_std)
    print("\n  Total mean ms (sampling + reduce):")
    for label, bench in all_runs:
        if bench:
            total = _total_ms(bench)
            speedup = total_std / total if total > 0 else float("nan")
            print(f"    {label:<16}: {total:7.3f} ms   ({speedup:.2f}x vs V1)")
        else:
            print(f"    {label:<16}: SKIPPED")

    # ── V8 Map-Centric (synthetic overlapping habitats) ──────────
    _run_map_centric_demo(
        n_runs=int(os.getenv("MAP_CENTRIC_RUNS", "100000")),
        seed=mc_config.seed,
        n_habitats=int(os.getenv("MAP_CENTRIC_HABITATS", "80")),
    )


if __name__ == "__main__":
    main()
