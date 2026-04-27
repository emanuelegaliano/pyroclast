"""PyOpenCL 2-D Monte Carlo Adapter — GPU implementation of IMonteCarloAdapter.

This module is the **only** file in the 2-D Monte Carlo pipeline that imports
``pyopencl``.  All other modules depend on the abstract
:class:`~pyroclast.ABCs.monte_carlo.IMonteCarloAdapter` Port.

Architectural role
------------------
``PyOpenCLMonteCarloAdapter2D`` is a *secondary adapter* (driven adapter).
It translates the domain-level Monte Carlo request into a two-kernel PyOpenCL
pipeline against ``pyroclast/kernels/monte_carlo_2d.cl``.

Algorithm
---------
Unlike the 1-D adapter (one work-item per run), this adapter launches a 2-D
NDRange where each work-item executes exactly **one random draw** for a
``(run, cell)`` pair:

Kernel 1 — ``mc_partial_sums`` (2-D NDRange, work-group ``WG_R × WG_C``):
  Each work-item ``(r, k)`` seeds MWC64X at ``base_offset + r*n_cells + k``,
  draws ``x ~ U(0,1)``, and tests ``x <= p_vec[k]``.  A tree reduction along
  the cells axis collapses the ``WG_C`` bits per run-row into a partial
  invaded count written to the intermediate buffer ``g_partial``.

Kernel 2 — ``mc_threshold_count`` (1-D NDRange, work-group ``WG_R*WG_C``):
  Each work-item ``r`` sums its ``n_groups_c`` partial counts, applies the
  threshold test, and participates in a tree reduction.  Thread 0 of each
  work-group issues one ``atomic_add`` to the global success counter.

Memory layout
-------------
* ``p_vec`` buffer: ``n_cells × sizeof(float32)`` — READ_ONLY.
  Kept in VRAM for the full duration of ``run_batched()``; released in
  a ``finally`` block after all batches complete.
* ``g_partial`` buffer: ``n_runs × n_groups_c × sizeof(int32)`` — READ_WRITE,
  initialised to 0 before each kernel-1 launch.
* ``count`` buffer: ``1 × sizeof(int32)`` — READ_WRITE, initialised to 0.

See also
--------
pyroclast.ABCs.monte_carlo.IMonteCarloAdapter : the Port this class implements.
pyroclast.kernels.monte_carlo_2d.cl : the OpenCL kernel source.
pyroclast.adapters.opencl_mc_adapter.PyOpenCLMonteCarloAdapter : 1-D variant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.ABCs.monte_carlo import IMonteCarloAdapter
from pyroclast.domain.models import BenchResult, CompactedHabitat, MonteCarloConfig

logger = logging.getLogger(__name__)

_KERNEL_PARTIAL = "mc_partial_sums"
_KERNEL_COUNT   = "mc_threshold_count"


def _find_gpu_device() -> cl.Device | None:
    try:
        for platform in cl.get_platforms():
            gpu_devices = platform.get_devices(cl.device_type.GPU)
            if gpu_devices:
                return gpu_devices[0]
    except cl.Error as exc:
        logger.warning("OpenCL platform enumeration failed: %s", exc)
    return None


def _build_context() -> cl.Context:
    gpu = _find_gpu_device()
    if gpu is not None:
        logger.info(
            "PyOpenCLMonteCarloAdapter2D: using GPU '%s' on platform '%s'.",
            gpu.name,
            gpu.platform.name,
        )
        return cl.Context(devices=[gpu])
    logger.warning(
        "PyOpenCLMonteCarloAdapter2D: no GPU found — falling back to "
        "create_some_context()."
    )
    return cl.create_some_context(interactive=False)


class PyOpenCLMonteCarloAdapter2D(IMonteCarloAdapter):
    """GPU Monte Carlo adapter using a 2-D NDRange via PyOpenCL.

    Each work-item draws a single random sample for one ``(run, cell)`` pair.
    Two sequential kernels perform the necessary reductions:
    ``mc_partial_sums`` collapses the cells axis per run, and
    ``mc_threshold_count`` applies the threshold and counts successes.

    Construction performs three one-time operations:

    1. **Device discovery** — selects the first GPU (or any device as fallback).
    2. **Context and queue creation** — initialises the OpenCL runtime.
    3. **Kernel compilation** — reads ``monte_carlo_2d.cl`` and builds both
       kernels, injecting ``WG_R`` and ``WG_C`` as ``-D`` preprocessor flags.

    Parameters
    ----------
    kernel_path : pathlib.Path, optional
        Path to ``monte_carlo_2d.cl``.  Defaults to the bundled kernel.
    profiling : bool, optional
        Enable OpenCL command-queue profiling.  Required for ``benchmark()``.
    wg_r : int, optional
        Work-group size along the runs dimension.  Default: 8.
    wg_c : int, optional
        Work-group size along the cells dimension.  Default: 32.
        ``wg_r * wg_c`` must equal 256.

    Raises
    ------
    FileNotFoundError
        If ``kernel_path`` does not exist.
    ValueError
        If ``wg_r * wg_c != 256``.
    RuntimeError
        If OpenCL kernel compilation fails.
    pyopencl.Error
        If no OpenCL platform or device is available.

    Examples
    --------
    >>> adapter = PyOpenCLMonteCarloAdapter2D()
    >>> prob = adapter.run(habitat, config)
    """

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        wg_r: int = 8,
        wg_c: int = 32,
    ) -> None:
        if wg_r * wg_c != 256:
            raise ValueError(
                f"wg_r * wg_c must equal 256, got {wg_r} * {wg_c} = {wg_r * wg_c}."
            )
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo_2d.cl"
            )
        kernel_path = Path(kernel_path)
        if not kernel_path.is_file():
            raise FileNotFoundError(
                f"OpenCL kernel not found at: {kernel_path}"
            )

        self._wg_r = wg_r
        self._wg_c = wg_c

        self._ctx: cl.Context = _build_context()
        self._profiling = profiling
        queue_props = (
            cl.command_queue_properties.PROFILING_ENABLE if profiling else 0
        )
        self._queue: cl.CommandQueue = cl.CommandQueue(
            self._ctx, properties=queue_props
        )
        self._kernel_launches_partial: list[tuple[float, int]] = []
        self._kernel_launches_count:   list[tuple[float, int]] = []
        self._last_n_cells: int = 0

        mwc64x_include = (
            Path(__file__).parent.parent.parent / "mwc64x-v0" / "mwc64x" / "cl"
        )
        kernel_source = kernel_path.read_text(encoding="utf-8")
        build_opts = f"-I {mwc64x_include} -D WG_R={wg_r} -D WG_C={wg_c}"
        try:
            self._program: cl.Program = cl.Program(
                self._ctx, kernel_source
            ).build(options=build_opts)
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL kernel compilation failed.\n"
                f"Kernel path: {kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        self._k_partial: cl.Kernel = cl.Kernel(self._program, _KERNEL_PARTIAL)
        self._k_count:   cl.Kernel = cl.Kernel(self._program, _KERNEL_COUNT)
        logger.info(
            "PyOpenCLMonteCarloAdapter2D: kernels '%s' + '%s' compiled "
            "(WG_R=%d, WG_C=%d).",
            _KERNEL_PARTIAL, _KERNEL_COUNT, wg_r, wg_c,
        )

    # ── IMonteCarloAdapter ────────────────────────────────────────────────────

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate destruction probability for a single habitat via GPU.

        Parameters
        ----------
        habitat : CompactedHabitat
        config : MonteCarloConfig

        Returns
        -------
        float
            Estimated probability in ``[0.0, 1.0]``.
        """
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)
        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            count = self._run_kernels(
                p_buf,
                habitat.n_cells,
                config.n_runs,
                config.threshold,
                np.uint64(int(config.seed)),
            )
        finally:
            if p_buf is not None:
                p_buf.release()

        prob = count / config.n_runs
        logger.debug(
            "PyOpenCLMonteCarloAdapter2D: habitat '%s' — prob=%.4f "
            "(R=%d, N_c=%d, theta=%.3f).",
            habitat.habitat_code, prob,
            config.n_runs, habitat.n_cells, config.threshold,
        )
        return prob

    def run_batched(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
        n_batches: int,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> float:
        """Estimate destruction probability using n_batches kernel launches.

        Uploads ``habitat.p_vec`` once and keeps it in VRAM for all batches.

        Parameters
        ----------
        habitat : CompactedHabitat
        config : MonteCarloConfig
            ``config.n_runs`` must be divisible by ``n_batches``.
        n_batches : int
        callback : callable, optional

        Returns
        -------
        float
        """
        if config.n_runs % n_batches != 0:
            raise ValueError(
                f"config.n_runs ({config.n_runs}) must be divisible by "
                f"n_batches ({n_batches})."
            )

        batch_size = config.n_runs // n_batches
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)
        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            total_count = 0
            for i in range(n_batches):
                base_offset = np.uint64(
                    int(config.seed) + i * batch_size * habitat.n_cells
                )
                total_count += self._run_kernels(
                    p_buf,
                    habitat.n_cells,
                    batch_size,
                    config.threshold,
                    base_offset,
                )
                if callback is not None:
                    runs_so_far = (i + 1) * batch_size
                    callback(i, n_batches, total_count / runs_so_far)
        finally:
            if p_buf is not None:
                p_buf.release()

        prob = total_count / config.n_runs
        logger.debug(
            "PyOpenCLMonteCarloAdapter2D.run_batched: habitat '%s' — prob=%.4f "
            "(R=%d, batches=%d, N_c=%d, theta=%.3f).",
            habitat.habitat_code, prob,
            config.n_runs, n_batches, habitat.n_cells, config.threshold,
        )
        return prob

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_kernels(
        self,
        p_buf: cl.Buffer,
        n_cells: int,
        n_runs: int,
        threshold: float,
        base_offset: np.uint64,
    ) -> int:
        """Launch both kernels and return the raw success count."""
        wg_r    = self._wg_r
        wg_c    = self._wg_c
        wg_size = wg_r * wg_c   # 256

        n_groups_c     = (n_cells + wg_c - 1) // wg_c
        n_runs_padded  = ((n_runs  + wg_r    - 1) // wg_r)    * wg_r
        n_cells_padded = n_groups_c * wg_c
        padded_count   = ((n_runs  + wg_size - 1) // wg_size) * wg_size

        # Transposed layout: [n_groups_c][n_runs_padded] — coalesced access.
        partial_host = np.zeros(n_groups_c * n_runs_padded, dtype=np.int32)
        count_host   = np.zeros(1, dtype=np.int32)

        mf = cl.mem_flags
        partial_buf: cl.Buffer | None = None
        count_buf:   cl.Buffer | None = None
        try:
            partial_buf = cl.Buffer(
                self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=partial_host
            )
            count_buf = cl.Buffer(
                self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=count_host
            )

            event1 = self._k_partial(
                self._queue,
                (n_runs_padded, n_cells_padded),
                (wg_r, wg_c),
                p_buf,
                partial_buf,
                np.uint32(n_cells),
                np.uint32(n_runs),
                base_offset,
                np.uint32(n_runs_padded),
            )

            event2 = self._k_count(
                self._queue,
                (padded_count,),
                (wg_size,),
                partial_buf,
                count_buf,
                np.uint32(n_cells),
                np.float32(threshold),
                np.uint32(n_runs),
                np.uint32(n_groups_c),
                np.uint32(n_runs_padded),
            )

            cl.enqueue_copy(self._queue, count_host, count_buf)
            self._queue.finish()

            if self._profiling:
                self._last_n_cells = n_cells
                self._kernel_launches_partial.append((
                    (event1.profile.end - event1.profile.start) * 1e-6,
                    4 * n_cells * n_runs,
                ))
                self._kernel_launches_count.append((
                    (event2.profile.end - event2.profile.start) * 1e-6,
                    4 * n_groups_c * n_runs_padded,
                ))
        finally:
            if partial_buf is not None:
                partial_buf.release()
            if count_buf is not None:
                count_buf.release()

        return int(count_host[0])

    # ── Profiling ─────────────────────────────────────────────────────────────

    def reset_profile(self) -> None:
        """Clear accumulated kernel timing data."""
        self._kernel_launches_partial.clear()
        self._kernel_launches_count.clear()
        self._last_n_cells = 0

    def benchmark(self) -> list[BenchResult]:
        """Return timing and bandwidth statistics from real kernel executions.

        Raises
        ------
        NotImplementedError
            If constructed without ``profiling=True``.
        ValueError
            If no kernel launches have been recorded yet.
        """
        if not self._profiling:
            raise NotImplementedError(
                "Profiling is disabled. Construct with profiling=True."
            )
        if not self._kernel_launches_partial:
            raise ValueError(
                "No kernel executions recorded yet. "
                "Call run() or run_batched() at least once before benchmark()."
            )

        def _make_bench(
            name: str, launches: list[tuple[float, int]]
        ) -> BenchResult:
            times_ms     = [t for t, _ in launches]
            total_bytes  = sum(b for _, b in launches)
            total_time_s = sum(times_ms) * 1e-3
            return BenchResult(
                kernel_name=name,
                shape=(self._last_n_cells, 1),
                n_cells=self._last_n_cells,
                n_runs=len(times_ms),
                mean_ms=float(np.mean(times_ms)),
                min_ms=float(np.min(times_ms)),
                bandwidth_gbs=total_bytes / total_time_s / 1e9,
            )

        return [
            _make_bench(_KERNEL_PARTIAL, self._kernel_launches_partial),
            _make_bench(_KERNEL_COUNT,   self._kernel_launches_count),
        ]
