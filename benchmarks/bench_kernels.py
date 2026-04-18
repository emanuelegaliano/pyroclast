"""Kernel benchmarks: execution time and effective memory bandwidth.

Run standalone:
    python benchmarks/bench_kernels.py

Or import bench_map_multiply() from test.py to print results inline.

Bandwidth model for map_multiply:
    reads  : float32 p_map (4 B) + uchar h_map (1 B) = 5 B/cell
    writes : float32 out_map (4 B)                   = 4 B/cell
    total  : 9 B/cell per invocation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

_KERNEL_PATH = Path(__file__).parent.parent / "pyroclast" / "kernels" / "preprocessing.cl"
_BYTES_PER_CELL = 9  # 4 (p_map) + 1 (h_map) + 4 (out_map)


@dataclass
class BenchResult:
    kernel_name: str
    shape: tuple[int, int]
    n_cells: int
    n_runs: int
    mean_ms: float
    min_ms: float
    bandwidth_gbs: float


def bench_map_multiply(
    shape: tuple[int, int] = (4096, 4096),
    n_warmup: int = 5,
    n_runs: int = 20,
) -> BenchResult:
    """Benchmark map_multiply using OpenCL profiling events.

    Measures only kernel execution time.
    Buffers are allocated once and reused across all runs.
    """
    n_cells = shape[0] * shape[1]

    # Device setup
    ctx: cl.Context | None = None
    for platform in cl.get_platforms():
        gpus = platform.get_devices(cl.device_type.GPU)
        if gpus:
            ctx = cl.Context(devices=[gpus[0]])
            break
    if ctx is None:
        ctx = cl.create_some_context(interactive=False)

    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    program = cl.Program(ctx, _KERNEL_PATH.read_text(encoding="utf-8")).build()
    kernel = cl.Kernel(program, "map_multiply")

    # Host buffers (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    p_flat = rng.random(n_cells).astype(np.float32)
    h_flat = rng.integers(0, 2, size=n_cells, dtype=np.uint8)

    mf = cl.mem_flags
    p_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_flat)
    h_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_flat)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=p_flat.nbytes)

    def _run_once() -> float:
        event = kernel(queue, (n_cells,), None, p_buf, h_buf, out_buf, np.int32(n_cells))
        event.wait()
        return (event.profile.end - event.profile.start) * 1e-6  # ns → ms

    for _ in range(n_warmup):
        _run_once()

    times_ms = [_run_once() for _ in range(n_runs)]

    p_buf.release()
    h_buf.release()
    out_buf.release()

    mean_ms = float(np.mean(times_ms))
    min_ms = float(np.min(times_ms))
    bandwidth_gbs = (_BYTES_PER_CELL * n_cells) / (mean_ms * 1e-3) / 1e9

    return BenchResult(
        kernel_name="map_multiply",
        shape=shape,
        n_cells=n_cells,
        n_runs=n_runs,
        mean_ms=mean_ms,
        min_ms=min_ms,
        bandwidth_gbs=bandwidth_gbs,
    )


def _print_result(r: BenchResult) -> None:
    print(f"  kernel      : {r.kernel_name}")
    print(f"  shape       : {r.shape[0]} x {r.shape[1]}  ({r.n_cells:,} cells)")
    print(f"  runs        : {r.n_runs}")
    print(f"  time (mean) : {r.mean_ms:.3f} ms")
    print(f"  time (min)  : {r.min_ms:.3f} ms")
    print(f"  bandwidth   : {r.bandwidth_gbs:.2f} GB/s  "
          f"(model: {_BYTES_PER_CELL} B/cell × {r.n_cells:,} cells)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Kernel benchmarks")
    print("=" * 60)
    result = bench_map_multiply()
    _print_result(result)
    print()
