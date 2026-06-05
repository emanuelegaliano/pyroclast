"""Unit test for the shared ``reduce_sum_int`` kernel.

The kernel is the core of the recursive reduction used by every Monte
Carlo adapter. We exercise it directly (outside the MC pipeline) on a
range of input sizes to make sure:

* the sliding-window guard handles ``n_elems < REDUCE_WG_SIZE`` correctly,
* the recursion converges to a single int for arbitrary ``n_elems``,
* the result equals ``int(numpy.sum(input))``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyopencl as cl
import pytest

from pyroclast.adapters.opencl_mc_adapter import _REDUCE_LWS, _build_context


@pytest.fixture(scope="module")
def reduce_runtime():
    try:
        ctx = _build_context()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")
    queue = cl.CommandQueue(ctx)
    src_path = (
        Path(__file__).parent.parent
        / "pyroclast" / "kernels" / "reduction" / "reduce_sum.cl"
    )
    program = cl.Program(ctx, src_path.read_text(encoding="utf-8")).build(options=f"-DREDUCE_WG_SIZE={_REDUCE_LWS}")
    kernel = cl.Kernel(program, "reduce_sum_int")
    return ctx, queue, kernel


def _reduce(reduce_runtime, host_arr: np.ndarray) -> int:
    ctx, queue, kernel = reduce_runtime
    n = int(host_arr.size)
    mf = cl.mem_flags
    buf_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_arr)
    buf_b = cl.Buffer(ctx, mf.READ_WRITE, size=max(1, n) * 4)
    src, dst = buf_a, buf_b
    try:
        n_remaining = n
        while n_remaining > 1:
            n_wg = max(1, math.ceil(n_remaining / _REDUCE_LWS))
            gws = n_wg * _REDUCE_LWS
            kernel(
                queue,
                (gws,),
                (_REDUCE_LWS,),
                dst,
                src,
                cl.LocalMemory(4 * _REDUCE_LWS),
                np.uint32(n_remaining),
            )
            src, dst = dst, src
            n_remaining = n_wg
        out = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, out, src)
        queue.finish()
        return int(out[0])
    finally:
        buf_a.release()
        buf_b.release()


@pytest.mark.parametrize("n", [1, 100, 256, 1000, 1024, 65_536, 1_048_576])
def test_reduce_matches_numpy(reduce_runtime, n):
    rng = np.random.default_rng(seed=n)
    arr = rng.integers(low=-5, high=5, size=n, dtype=np.int32)
    expected = int(arr.sum())
    actual = _reduce(reduce_runtime, arr.copy())
    assert actual == expected


def test_reduce_constant_array(reduce_runtime):
    arr = np.ones(10_000, dtype=np.int32)
    assert _reduce(reduce_runtime, arr.copy()) == 10_000


def test_reduce_with_negative_values(reduce_runtime):
    arr = np.full(2000, -3, dtype=np.int32)
    assert _reduce(reduce_runtime, arr.copy()) == -6000
