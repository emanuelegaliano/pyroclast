import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    InvasionCriteria,
    PyOpenCLAdapter,
)
from pyroclast.services import run_preprocessing_batch
from benchmarks.bench_kernels import _print_result, bench_map_multiply


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip('/"\'')
    force_recompute = os.getenv("FORCE_RECOMPUTE", "0").strip().lower() in ("1", "true", "yes")
    cache_dir = Path(data_path) / "cache"  # Caching on the same volume as input
    cache_dir.mkdir(exist_ok=True)

    # ── 1. Repository ────────────────────────────────────────────
    section("1. Repository — raw raster data")
    repo = FileMapRepository(data_path)

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
    adapter = PyOpenCLAdapter()
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

    # ── 6. Kernel benchmark ──────────────────────────────────────
    section("6. Kernel benchmark — map_multiply")
    n_rows, n_cols = invasion.data.shape
    bench = bench_map_multiply(shape=(n_rows, n_cols), n_warmup=5, n_runs=20)
    _print_result(bench)


if __name__ == "__main__":
    main()
