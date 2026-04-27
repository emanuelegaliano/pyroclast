"""PyOpenCL Monte Carlo Adapter — GPU implementation of IMonteCarloAdapter.

This module is the **only** file in the Monte Carlo pipeline that imports
``pyopencl``.  All other modules depend on the abstract
:class:`~pyroclast.ABCs.monte_carlo.IMonteCarloAdapter` Port.

Architectural role
------------------
``PyOpenCLMonteCarloAdapter`` is a *secondary adapter* (driven adapter).
It translates the domain-level Monte Carlo request into low-level PyOpenCL
API calls against ``pyroclast/kernels/monte_carlo.cl``.

Algorithm (Kernel 2)
--------------------
The kernel uses a 1-D NDRange of ``config.n_runs`` work-items.  Each
work-item ``r``:

1. Seeds an MWC64X RNG stream at position ``base_offset + r * n_cells``.
2. Loops over the ``habitat.n_cells`` compacted habitat cells.
3. For each cell ``k`` draws ``x ~ U(0, 1)`` and tests ``x <= p_vec[k]``.
4. Checks whether ``invaded_fraction > config.threshold``.
5. Contributes ``1`` or ``0`` to a work-group reduction via
   ``work_group_reduce_add``; thread 0 of each group atomically adds the
   group sum to a single global counter.

The host reads back the counter (1 × int32 = 4 bytes) and computes
``prob = count / n_runs``.

Memory layout
-------------
* ``p_vec`` buffer: ``n_cells × sizeof(float32)`` — READ_ONLY.
  In ``run()`` it is allocated and released per call.
  In ``run_batched()`` it is allocated once before the batch loop and
  released in a ``finally`` block after all batches complete.
* ``count`` buffer: ``1 × sizeof(int32)`` — READ_WRITE, initialised to 0
  by the host before each kernel launch.

Device selection
----------------
Identical strategy to :class:`~pyroclast.adapters.opencl_adapter.PyOpenCLAdapter`:
first GPU found, fallback to any available device via
``pyopencl.create_some_context``.

OpenCL version
--------------
Requires OpenCL 1.2 or later.  The kernel is compiled with a ``-I`` flag
pointing to the MWC64X header directory.

See also
--------
pyroclast.ABCs.monte_carlo.IMonteCarloAdapter : the Port this class implements.
pyroclast.kernels.monte_carlo.cl : the OpenCL kernel source.
pyroclast.services.monte_carlo.run_monte_carlo : primary consumer.
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

_KERNEL_NAME = "monte_carlo_run"


def _find_gpu_device() -> cl.Device | None:
    """Scan all OpenCL platforms and return the first GPU device found.

    Returns
    -------
    pyopencl.Device or None
        The first ``CL_DEVICE_TYPE_GPU`` device, or ``None`` if unavailable.
    """
    try:
        for platform in cl.get_platforms():
            gpu_devices = platform.get_devices(cl.device_type.GPU)
            if gpu_devices:
                return gpu_devices[0]
    except cl.Error as exc:
        logger.warning("OpenCL platform enumeration failed: %s", exc)
    return None


def _build_context() -> cl.Context:
    """Construct an OpenCL context, preferring a GPU device.

    Returns
    -------
    pyopencl.Context
        A valid OpenCL context ready for use.

    Raises
    ------
    pyopencl.Error
        If context creation fails on all available devices.
    """
    gpu = _find_gpu_device()
    if gpu is not None:
        logger.info(
            "PyOpenCLMonteCarloAdapter: using GPU '%s' on platform '%s'.",
            gpu.name,
            gpu.platform.name,
        )
        return cl.Context(devices=[gpu])
    logger.warning(
        "PyOpenCLMonteCarloAdapter: no GPU found — falling back to "
        "create_some_context()."
    )
    return cl.create_some_context(interactive=False)


class PyOpenCLMonteCarloAdapter(IMonteCarloAdapter):
    """GPU Monte Carlo adapter implementing IMonteCarloAdapter via PyOpenCL.

    This adapter fulfils the
    :class:`~pyroclast.ABCs.monte_carlo.IMonteCarloAdapter` contract using
    the ``monte_carlo_run`` OpenCL kernel.  It is constructed once and can
    be reused across multiple ``run`` calls.

    Construction performs three one-time operations:

    1. **Device discovery** — selects the first GPU (or any device as
       fallback).
    2. **Context and queue creation** — initialises the OpenCL runtime.
    3. **Kernel compilation** — reads ``monte_carlo.cl`` and builds the
       ``monte_carlo_run`` kernel with a ``-I`` flag pointing at the
       MWC64X header directory.

    Parameters
    ----------
    kernel_path : pathlib.Path, optional
        Path to the OpenCL kernel source file.  Defaults to
        ``pyroclast/kernels/monte_carlo.cl`` resolved relative to this
        module.  Override in tests to use a stub kernel.

    Raises
    ------
    FileNotFoundError
        If ``kernel_path`` does not exist.
    RuntimeError
        If the OpenCL kernel fails to compile.
    pyopencl.Error
        If no OpenCL platform or device is available.

    Examples
    --------
    >>> from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
    >>> adapter = PyOpenCLMonteCarloAdapter()
    >>> prob = adapter.run(habitat, config)
    """

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo.cl"
            )
        kernel_path = Path(kernel_path)
        if not kernel_path.is_file():
            raise FileNotFoundError(
                f"OpenCL kernel not found at: {kernel_path}"
            )

        self._ctx: cl.Context = _build_context()
        self._profiling = profiling
        queue_props = (
            cl.command_queue_properties.PROFILING_ENABLE if profiling else 0
        )
        self._queue: cl.CommandQueue = cl.CommandQueue(
            self._ctx, properties=queue_props
        )
        # each entry: (elapsed_ms, bytes_transferred) for one kernel launch
        self._kernel_launches: list[tuple[float, int]] = []
        self._last_n_cells: int = 0

        mwc64x_include = (
            Path(__file__).parent.parent.parent / "mwc64x-v0" / "mwc64x" / "cl"
        )
        kernel_source = kernel_path.read_text(encoding="utf-8")
        try:
            self._program: cl.Program = cl.Program(
                self._ctx, kernel_source
            ).build(options=f"-I {mwc64x_include}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL kernel compilation failed.\n"
                f"Kernel path: {kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        self._kernel: cl.Kernel = cl.Kernel(self._program, _KERNEL_NAME)
        logger.info(
            "PyOpenCLMonteCarloAdapter: kernel '%s' compiled successfully.",
            _KERNEL_NAME,
        )

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate destruction probability for a single habitat via GPU.

        Transfers ``habitat.p_vec`` to VRAM, launches ``monte_carlo_run``
        with a 1-D NDRange of ``config.n_runs`` work-items, reads back a
        single int32 counter, and computes the probability estimate.

        Parameters
        ----------
        habitat : CompactedHabitat
            Pre-processed habitat.  ``habitat.n_cells`` must be > 0.
        config : MonteCarloConfig
            Simulation parameters.

        Returns
        -------
        float
            Estimated probability in ``[0.0, 1.0]``.

        Raises
        ------
        pyopencl.Error
            On any OpenCL runtime error.
        """
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)
        count_host = np.zeros(1, dtype=np.int32)

        wg = 256
        padded = ((config.n_runs + wg - 1) // wg) * wg

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        count_buf: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=p_host,
            )
            count_buf = cl.Buffer(
                self._ctx,
                mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=count_host,
            )

            event = self._kernel(
                self._queue,
                (padded,),
                (wg,),
                p_buf,
                count_buf,
                np.uint32(habitat.n_cells),
                np.float32(config.threshold),
                np.uint64(int(config.seed)),
                np.uint32(config.n_runs),
            )

            cl.enqueue_copy(self._queue, count_host, count_buf)
            self._queue.finish()

            if self._profiling:
                elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                # each of config.n_runs work-items reads all habitat.n_cells floats
                bytes_transferred = 4 * habitat.n_cells * config.n_runs
                self._last_n_cells = habitat.n_cells
                self._kernel_launches.append((elapsed_ms, bytes_transferred))
        finally:
            if p_buf is not None:
                p_buf.release()
            if count_buf is not None:
                count_buf.release()

        prob = int(count_host[0]) / config.n_runs
        logger.debug(
            "PyOpenCLMonteCarloAdapter: habitat '%s' — prob=%.4f "
            "(R=%d, N_c=%d, theta=%.3f).",
            habitat.habitat_code,
            prob,
            config.n_runs,
            habitat.n_cells,
            config.threshold,
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

        Uploads ``habitat.p_vec`` once and keeps it in VRAM for all batches,
        downloading only a single int32 per batch.

        Parameters
        ----------
        habitat : CompactedHabitat
            Pre-processed habitat.  ``habitat.n_cells`` must be > 0.
        config : MonteCarloConfig
            Simulation parameters.  ``config.n_runs`` must be divisible by
            ``n_batches``.
        n_batches : int
            Number of kernel launches.
        callback : callable, optional
            Called after each batch as ``callback(batch_index, n_batches,
            partial_prob)``.

        Returns
        -------
        float
            Estimated probability in ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If ``config.n_runs`` is not divisible by ``n_batches``.
        pyopencl.Error
            On any OpenCL runtime error.
        """
        if config.n_runs % n_batches != 0:
            raise ValueError(
                f"config.n_runs ({config.n_runs}) must be divisible by "
                f"n_batches ({n_batches})."
            )

        batch_size = config.n_runs // n_batches
        wg = 256
        padded_batch = ((batch_size + wg - 1) // wg) * wg
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=p_host,
            )

            total_count = 0
            for i in range(n_batches):
                count_host = np.zeros(1, dtype=np.int32)
                count_buf = cl.Buffer(
                    self._ctx,
                    mf.READ_WRITE | mf.COPY_HOST_PTR,
                    hostbuf=count_host,
                )
                try:
                    event = self._kernel(
                        self._queue,
                        (padded_batch,),
                        (wg,),
                        p_buf,
                        count_buf,
                        np.uint32(habitat.n_cells),
                        np.float32(config.threshold),
                        np.uint64(int(config.seed) + i * batch_size * habitat.n_cells),
                        np.uint32(batch_size),
                    )
                    cl.enqueue_copy(self._queue, count_host, count_buf)
                    self._queue.finish()

                    if self._profiling:
                        elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                        bytes_transferred = 4 * habitat.n_cells * batch_size
                        self._last_n_cells = habitat.n_cells
                        self._kernel_launches.append((elapsed_ms, bytes_transferred))
                finally:
                    count_buf.release()

                total_count += int(count_host[0])

                if callback is not None:
                    runs_so_far = (i + 1) * batch_size
                    callback(i, n_batches, total_count / runs_so_far)
        finally:
            if p_buf is not None:
                p_buf.release()

        prob = total_count / config.n_runs
        logger.debug(
            "PyOpenCLMonteCarloAdapter.run_batched: habitat '%s' — prob=%.4f "
            "(R=%d, batches=%d, N_c=%d, theta=%.3f).",
            habitat.habitat_code,
            prob,
            config.n_runs,
            n_batches,
            habitat.n_cells,
            config.threshold,
        )
        return prob

    def reset_profile(self) -> None:
        """Clear accumulated kernel timing data."""
        self._kernel_launches.clear()
        self._last_n_cells = 0

    def benchmark(self) -> list[BenchResult]:
        """Return timing and bandwidth statistics from real kernel executions.

        Raises
        ------
        NotImplementedError
            If the adapter was constructed without ``profiling=True``.
        ValueError
            If no kernel launches have been recorded yet.
        """
        if not self._profiling:
            raise NotImplementedError(
                "Profiling is disabled. Construct with profiling=True."
            )
        if not self._kernel_launches:
            raise ValueError(
                "No kernel executions recorded yet. "
                "Call run() or run_batched() at least once before benchmark()."
            )
        times_ms = [t for t, _ in self._kernel_launches]
        total_bytes = sum(b for _, b in self._kernel_launches)
        total_time_s = sum(times_ms) * 1e-3
        bandwidth_gbs = total_bytes / total_time_s / 1e9
        return [BenchResult(
            kernel_name=_KERNEL_NAME,
            shape=(self._last_n_cells, 1),
            n_cells=self._last_n_cells,
            n_runs=len(times_ms),
            mean_ms=float(np.mean(times_ms)),
            min_ms=float(np.min(times_ms)),
            bandwidth_gbs=bandwidth_gbs,
        )]
