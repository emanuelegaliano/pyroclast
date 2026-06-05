"""PyOpenCL Monte Carlo Adapter — GPU implementation of IMonteCarloAdapter.

This module is the **only** file in the Monte Carlo pipeline that imports
``pyopencl``.  All other modules depend on the abstract
:class:`~pyroclast.ABCs.monte_carlo.IMonteCarloAdapter` Port.

Architectural role
------------------
``PyOpenCLMonteCarloAdapter`` is a *secondary adapter* (driven adapter).
It translates the domain-level Monte Carlo request into low-level PyOpenCL
API calls against ``pyroclast/kernels/monte_carlo.cl`` (sampling) and
``pyroclast/kernels/reduce_sum.cl`` (recursive reduction).

Algorithm
---------
The sampling kernel writes one ``int`` per work-group into
``partial[group_id]`` — no atomics. The host then re-launches the shared
``reduce_sum_int`` kernel, ping-ponging two global buffers, until a
single scalar remains. Final read-back is one ``int32`` (4 bytes).

Memory layout
-------------
* ``p_vec`` buffer: ``n_cells × sizeof(float32)`` — READ_ONLY.
* ``partial_a`` and ``partial_b`` buffers: ``n_wg × sizeof(int32)`` each,
  used as ping-pong source/destination during the recursive reduce.

See also
--------
pyroclast.ABCs.monte_carlo.IMonteCarloAdapter : the Port this class implements.
pyroclast.kernels.monte_carlo.cl : the OpenCL sampling kernel source.
pyroclast.kernels.reduce_sum.cl : the OpenCL recursive reducer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.ABCs.monte_carlo import IMonteCarloAdapter
from pyroclast.domain.models import BenchResult, CompactedHabitat, GridTopology, MonteCarloConfig

logger = logging.getLogger(__name__)

_REDUCE_KERNEL_NAME = "reduce_sum_int"
_REDUCE_LWS = 256  # matches REDUCE_WG_SIZE in reduce_sum.cl


def _find_gpu_device() -> cl.Device | None:
    """Scan all OpenCL platforms and return the first GPU device found."""
    try:
        for platform in cl.get_platforms():
            gpu_devices = platform.get_devices(cl.device_type.GPU)
            if gpu_devices:
                return gpu_devices[0]
    except cl.Error as exc:
        logger.warning("OpenCL platform enumeration failed: %s", exc)
    return None


def _build_context() -> cl.Context:
    """Construct an OpenCL context, preferring a GPU device."""
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
    """GPU Monte Carlo adapter using a sampling kernel + recursive reducer.

    Construction performs four one-time operations:

    1. Device discovery (first GPU, fallback to ``create_some_context``).
    2. Context and queue creation.
    3. Compilation of the sampling kernel (``monte_carlo.cl`` by default).
    4. Compilation of the shared ``reduce_sum.cl`` reducer.

    Subclasses with a different OpenCL kernel function name should set
    ``_SAMPLING_KERNEL_NAME`` accordingly.
    """

    _SAMPLING_KERNEL_NAME: str = "monte_carlo_run"
    _MULTI_SAMPLING_KERNEL_NAME: str = "monte_carlo_run_multi"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        extra_build_options: str = "",
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo" / "monte_carlo.cl"
            )
        kernel_path = Path(kernel_path)
        if not kernel_path.is_file():
            raise FileNotFoundError(
                f"OpenCL kernel not found at: {kernel_path}"
            )

        reduce_kernel_path = (
            Path(__file__).parent.parent / "kernels" / "reduction" / "reduce_sum.cl"
        )
        if not reduce_kernel_path.is_file():
            raise FileNotFoundError(
                f"OpenCL reducer kernel not found at: {reduce_kernel_path}"
            )

        self._ctx: cl.Context = _build_context()
        self._profiling = profiling
        queue_props = (
            cl.command_queue_properties.PROFILING_ENABLE if profiling else 0
        )
        self._queue: cl.CommandQueue = cl.CommandQueue(
            self._ctx, properties=queue_props
        )

        # Per-call profiling: sampling kernel and reduce passes are tracked
        # separately so benchmark() can break them down.
        self._kernel_launches: list[tuple[float, int]] = []
        self._reduce_launches: list[tuple[float, int]] = []
        self._last_n_cells: int = 0
        self._last_n_wg: int = 0
        self._last_kernel_name: str = self._SAMPLING_KERNEL_NAME

        self._kernel_path = kernel_path
        self._mwc64x_include = (
            Path(__file__).parent.parent.parent / "mwc64x-v0" / "mwc64x" / "cl"
        )
        self._kernel_dir = kernel_path.parent
        self._compiled_wg_size = 256
        # Extra -D defines appended by subclasses (e.g. -DVEC_WIDTH=4). Stored
        # so dynamic _recompile() preserves them across an LWS change.
        self._extra_build_options = extra_build_options

        kernel_source = kernel_path.read_text(encoding="utf-8")
        try:
            self._program: cl.Program = cl.Program(
                self._ctx, kernel_source
            ).build(options=f"-I {self._mwc64x_include} -I {self._kernel_dir} -DWG_SIZE={self._compiled_wg_size} {self._extra_build_options}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL kernel compilation failed.\n"
                f"Kernel path: {kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        reduce_source = reduce_kernel_path.read_text(encoding="utf-8")
        try:
            self._reduce_program: cl.Program = cl.Program(
                self._ctx, reduce_source
            ).build(options=f"-DREDUCE_WG_SIZE={_REDUCE_LWS}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL reducer kernel compilation failed.\n"
                f"Kernel path: {reduce_kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        self._kernel: cl.Kernel = cl.Kernel(
            self._program, self._SAMPLING_KERNEL_NAME
        )
        self._reduce_kernel: cl.Kernel = cl.Kernel(
            self._reduce_program, _REDUCE_KERNEL_NAME
        )
        logger.info(
            "PyOpenCLMonteCarloAdapter: kernels '%s' and '%s' compiled.",
            self._SAMPLING_KERNEL_NAME,
            _REDUCE_KERNEL_NAME,
        )

    def _recompile(self, wg_size: int) -> None:
        """Recompile the sampling kernel dynamically for a new workgroup size."""
        logger.info(
            "PyOpenCLMonteCarloAdapter: Recompiling kernel dynamically for WG_SIZE=%d",
            wg_size,
        )
        kernel_source = self._kernel_path.read_text(encoding="utf-8")
        try:
            self._program = cl.Program(
                self._ctx, kernel_source
            ).build(options=f"-I {self._mwc64x_include} -I {self._kernel_dir} -DWG_SIZE={wg_size} {self._extra_build_options}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL kernel dynamic recompilation failed.\n"
                f"Kernel path: {self._kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc
        self._kernel = cl.Kernel(
            self._program, self._SAMPLING_KERNEL_NAME
        )
        self._compiled_wg_size = wg_size

    def suggest_topology(self, n_runs: int) -> GridTopology:
        """Suggest an execution grid that saturates the device."""
        device = self._ctx.devices[0]
        max_cu = device.max_compute_units
        lws = 256
        gws = max_cu * 4 * lws
        return GridTopology(gws=int(gws), lws=int(lws))

    # ------------------------------------------------------------------
    # Recursive reducer
    # ------------------------------------------------------------------

    def _reduce_partial(
        self,
        partial_a: cl.Buffer,
        partial_b: cl.Buffer,
        n_elems: int,
    ) -> int:
        """Recursively reduce partial_a to a single int.

        Ping-pongs between ``partial_a`` and ``partial_b``, launching
        ``reduce_sum_int`` until one element remains, then copies it back
        to the host. Records timing/bandwidth in ``self._reduce_launches``
        when profiling is enabled.

        Parameters
        ----------
        partial_a : cl.Buffer
            Source buffer of ``n_elems`` ints (output of the sampling kernel).
        partial_b : cl.Buffer
            Scratch buffer of at least ``ceil(n_elems / _REDUCE_LWS)`` ints.
        n_elems : int
            Number of ints currently held in ``partial_a``.

        Returns
        -------
        int
            The scalar sum of the original ``partial_a[0..n_elems-1]``.
        """
        src, dst = partial_a, partial_b
        while n_elems > 1:
            n_wg = max(1, math.ceil(n_elems / _REDUCE_LWS))
            gws = n_wg * _REDUCE_LWS
            evt = self._reduce_kernel(
                self._queue,
                (gws,),
                (_REDUCE_LWS,),
                dst,
                src,
                cl.LocalMemory(4 * _REDUCE_LWS),
                np.uint32(n_elems),
            )
            if self._profiling:
                evt.wait()
                elapsed_ms = (evt.profile.end - evt.profile.start) * 1e-6
                total_bytes = (n_elems + n_wg) * 4
                self._reduce_launches.append((elapsed_ms, total_bytes))
            src, dst = dst, src
            n_elems = n_wg

        final = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(self._queue, final, src)
        self._queue.finish()
        return int(final[0])

    def _compile_multi_on_demand(self) -> None:
        """Compile the multi-habitat sampling and reduction kernels if not already done."""
        if hasattr(self, "_multi_program"):
            return

        stem = self._kernel_path.parent / f"{self._kernel_path.stem}_multi.cl"
        if not stem.is_file():
            raise FileNotFoundError(
                f"Multi-habitat OpenCL kernel not found at: {stem}"
            )

        reduce_multi_kernel_path = (
            Path(__file__).parent.parent / "kernels" / "reduction" / "reduce_sum_batched.cl"
        )
        if not reduce_multi_kernel_path.is_file():
            raise FileNotFoundError(
                f"OpenCL batched reducer kernel not found at: {reduce_multi_kernel_path}"
            )

        kernel_source = stem.read_text(encoding="utf-8")
        try:
            self._multi_program = cl.Program(
                self._ctx, kernel_source
            ).build(options=f"-I {self._mwc64x_include} -I {self._kernel_dir} -DWG_SIZE={self._compiled_wg_size} {self._extra_build_options}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL multi-habitat kernel compilation failed.\n"
                f"Kernel path: {stem}\n"
                f"Build log:\n{exc}"
            ) from exc

        reduce_source = reduce_multi_kernel_path.read_text(encoding="utf-8")
        try:
            self._reduce_multi_program = cl.Program(
                self._ctx, reduce_source
            ).build(options=f"-DREDUCE_WG_SIZE={_REDUCE_LWS}")
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL batched reducer kernel compilation failed.\n"
                f"Kernel path: {reduce_multi_kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        self._multi_kernel = cl.Kernel(
            self._multi_program, self._MULTI_SAMPLING_KERNEL_NAME
        )
        self._reduce_multi_kernel = cl.Kernel(
            self._reduce_multi_program, "reduce_sum_int_batched"
        )
        logger.info(
            "PyOpenCLMonteCarloAdapter: multi-habitat kernels '%s' and 'reduce_sum_int_batched' compiled.",
            self._MULTI_SAMPLING_KERNEL_NAME,
        )

    def _reduce_partial_batched(
        self,
        partial_a: cl.Buffer,
        partial_b: cl.Buffer,
        n_elems_in: int,
        n_habitats: int,
    ) -> np.ndarray:
        """Recursively reduce partial_a to N final ints, one per habitat."""
        src, dst = partial_a, partial_b
        while n_elems_in > 1:
            n_wg = max(1, math.ceil(n_elems_in / _REDUCE_LWS))
            gws_x = n_wg * _REDUCE_LWS
            gws = (gws_x, n_habitats)
            lws = (_REDUCE_LWS, 1)

            evt = self._reduce_multi_kernel(
                self._queue,
                gws,
                lws,
                dst,
                src,
                cl.LocalMemory(4 * _REDUCE_LWS),
                np.uint32(n_elems_in),
                np.uint32(n_habitats),
            )
            if self._profiling:
                evt.wait()
                elapsed_ms = (evt.profile.end - evt.profile.start) * 1e-6
                total_bytes = (n_elems_in + n_wg) * 4 * n_habitats
                self._reduce_launches.append((elapsed_ms, total_bytes))
            src, dst = dst, src
            n_elems_in = n_wg

        final = np.zeros(n_habitats, dtype=np.int32)
        cl.enqueue_copy(self._queue, final, src)
        self._queue.finish()
        return final

    # ------------------------------------------------------------------
    # IMonteCarloAdapter API
    # ------------------------------------------------------------------

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate destruction probability for a single habitat via GPU."""
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)

        topology = config.topology or self.suggest_topology(config.n_runs)
        gws = topology.gws
        lws = topology.lws
        if isinstance(lws, int) and lws != self._compiled_wg_size:
            self._recompile(lws)
        n_wg = gws // lws

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        partial_a: cl.Buffer | None = None
        partial_b: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=p_host,
            )
            partial_a = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=n_wg * 4
            )
            partial_b = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=max(1, n_wg) * 4
            )

            event = self._kernel(
                self._queue,
                (gws,),
                (lws,),
                p_buf,
                partial_a,
                np.uint32(habitat.n_cells),
                np.float32(config.threshold),
                np.uint64(int(config.seed)),
                np.uint32(config.n_runs),
            )

            if self._profiling:
                event.wait()
                elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                self._last_n_cells = habitat.n_cells
                self._last_n_wg = n_wg
                self._last_kernel_name = self._SAMPLING_KERNEL_NAME
                total_bytes = self.get_bytes_read(
                    habitat, config.n_runs
                ) + self.get_bytes_written(habitat, config.n_runs)
                self._kernel_launches.append((elapsed_ms, total_bytes))

            total_count = self._reduce_partial(partial_a, partial_b, n_wg)
        finally:
            for buf in (p_buf, partial_a, partial_b):
                if buf is not None:
                    buf.release()

        prob = total_count / config.n_runs
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
        """Estimate destruction probability using n_batches kernel launches."""
        if config.n_runs % n_batches != 0:
            raise ValueError(
                f"config.n_runs ({config.n_runs}) must be divisible by "
                f"n_batches ({n_batches})."
            )

        batch_size = config.n_runs // n_batches
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)

        topology = config.topology or self.suggest_topology(batch_size)
        gws = topology.gws
        lws = topology.lws
        if isinstance(lws, int) and lws != self._compiled_wg_size:
            self._recompile(lws)
        n_wg = gws // lws

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        partial_a: cl.Buffer | None = None
        partial_b: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=p_host,
            )
            partial_a = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=n_wg * 4
            )
            partial_b = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=max(1, n_wg) * 4
            )

            total_count = 0
            for i in range(n_batches):
                event = self._kernel(
                    self._queue,
                    (gws,),
                    (lws,),
                    p_buf,
                    partial_a,
                    np.uint32(habitat.n_cells),
                    np.float32(config.threshold),
                    np.uint64(int(config.seed) + i * batch_size * habitat.n_cells),
                    np.uint32(batch_size),
                )

                if self._profiling:
                    event.wait()
                    elapsed_ms = (
                        event.profile.end - event.profile.start
                    ) * 1e-6
                    self._last_n_cells = habitat.n_cells
                    self._last_n_wg = n_wg
                    self._last_kernel_name = self._SAMPLING_KERNEL_NAME
                    total_bytes = self.get_bytes_read(
                        habitat, batch_size
                    ) + self.get_bytes_written(habitat, batch_size)
                    self._kernel_launches.append((elapsed_ms, total_bytes))

                batch_count = self._reduce_partial(partial_a, partial_b, n_wg)
                total_count += batch_count

                if callback is not None:
                    runs_so_far = (i + 1) * batch_size
                    callback(i, n_batches, total_count / runs_so_far)
        finally:
            for buf in (p_buf, partial_a, partial_b):
                if buf is not None:
                    buf.release()

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

    def run_multi_habitats(
        self,
        habitats: list[CompactedHabitat],
        config: MonteCarloConfig,
    ) -> list[float]:
        """Estimate destruction probability for multiple habitats simultaneously via GPU."""
        if not habitats:
            return []

        self._compile_multi_on_demand()

        n_habitats = len(habitats)
        n_runs = config.n_runs

        p_stride = max(h.n_cells for h in habitats)
        if p_stride == 0:
            return [0.0] * n_habitats

        p_vecs_host = np.full(n_habitats * p_stride, -1.0, dtype=np.float32)
        n_cells_host = np.zeros(n_habitats, dtype=np.uint32)
        for h_idx, h in enumerate(habitats):
            n_cells_host[h_idx] = h.n_cells
            start = h_idx * p_stride
            p_vecs_host[start : start + h.n_cells] = np.asarray(h.p_vec, dtype=np.float32)

        topology = config.topology or self.suggest_topology(n_runs)
        gws_runs = topology.gws[0] if isinstance(topology.gws, tuple) else topology.gws
        lws_runs = topology.lws[0] if isinstance(topology.lws, tuple) else topology.lws

        if lws_runs != self._compiled_wg_size:
            self._recompile(lws_runs)
            if hasattr(self, "_multi_program"):
                delattr(self, "_multi_program")
            self._compile_multi_on_demand()

        n_wg_runs = gws_runs // lws_runs

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        n_cells_buf: cl.Buffer | None = None
        partial_a: cl.Buffer | None = None
        partial_b: cl.Buffer | None = None

        try:
            p_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=p_vecs_host,
            )
            n_cells_buf = cl.Buffer(
                self._ctx,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=n_cells_host,
            )
            partial_a = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=n_habitats * n_wg_runs * 4
            )
            partial_b = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=n_habitats * max(1, n_wg_runs) * 4
            )

            gws_sampling = (gws_runs,)
            lws_sampling = (lws_runs,)

            event = self._multi_kernel(
                self._queue,
                gws_sampling,
                lws_sampling,
                p_buf,
                n_cells_buf,
                partial_a,
                np.uint32(p_stride),
                np.float32(config.threshold),
                np.uint64(int(config.seed)),
                np.uint32(n_runs),
                np.uint32(n_habitats),
            )

            if self._profiling:
                event.wait()
                elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                self._last_n_cells = p_stride
                self._last_n_wg = n_wg_runs
                self._last_kernel_name = self._MULTI_SAMPLING_KERNEL_NAME
                total_bytes = (n_runs * sum(h.n_cells for h in habitats) * 4) + (n_habitats * n_wg_runs * 4)
                self._kernel_launches.append((elapsed_ms, total_bytes))

            total_counts = self._reduce_partial_batched(partial_a, partial_b, n_wg_runs, n_habitats)

        finally:
            for buf in (p_buf, n_cells_buf, partial_a, partial_b):
                if buf is not None:
                    buf.release()

        probs = [float(count) / n_runs for count in total_counts]
        return probs

    def reset_profile(self) -> None:
        """Clear accumulated kernel timing data."""
        self._kernel_launches.clear()
        self._reduce_launches.clear()
        self._last_n_cells = 0
        self._last_n_wg = 0

    def get_bytes_read(self, habitat: CompactedHabitat, n_runs: int) -> int:
        """Bytes read by the sampling kernel.

        Each trial iterates over all ``n_cells`` of ``p_vec`` inside
        ``_count_invaded``, so the kernel issues ``n_runs * n_cells``
        float32 loads. Every load counts at face value, regardless of
        cache hits.
        """
        return n_runs * habitat.n_cells * 4

    def get_bytes_written(self, habitat: CompactedHabitat, n_runs: int) -> int:
        """Bytes written by the sampling kernel.

        One int (4 B) per work-group is written to the partial buffer.
        Bytes written by the recursive reducer are tracked separately in
        ``self._reduce_launches``.
        """
        return self._last_n_wg * 4

    def benchmark(self) -> list[BenchResult]:
        """Return timing and bandwidth statistics for sampling + reduce.

        Returns
        -------
        list[BenchResult]
            One entry for the sampling kernel and (when reduce launches
            were recorded) one entry for the recursive reducer.
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

        results: list[BenchResult] = []

        sample_times = [t for t, _ in self._kernel_launches]
        sample_bytes = sum(b for _, b in self._kernel_launches)
        sample_time_s = sum(sample_times) * 1e-3
        sample_bw = sample_bytes / sample_time_s / 1e9 if sample_time_s > 0 else 0.0
        memory_mb = (self._last_n_cells * 4 + self._last_n_wg * 4 * 2) / 1e6
        results.append(BenchResult(
            kernel_name=self._last_kernel_name,
            shape=(self._last_n_cells, 1),
            n_cells=self._last_n_cells,
            n_runs=len(sample_times),
            mean_ms=float(np.mean(sample_times)),
            min_ms=float(np.min(sample_times)),
            bandwidth_gbs=sample_bw,
            memory_mb=memory_mb,
        ))

        if self._reduce_launches:
            r_times = [t for t, _ in self._reduce_launches]
            r_bytes = sum(b for _, b in self._reduce_launches)
            r_time_s = sum(r_times) * 1e-3
            r_bw = r_bytes / r_time_s / 1e9 if r_time_s > 0 else 0.0
            results.append(BenchResult(
                kernel_name=_REDUCE_KERNEL_NAME,
                shape=(self._last_n_wg, 1),
                n_cells=self._last_n_wg,
                n_runs=len(r_times),
                mean_ms=float(np.mean(r_times)),
                min_ms=float(np.min(r_times)),
                bandwidth_gbs=r_bw,
                memory_mb=self._last_n_wg * 4 * 2 / 1e6,
            ))

        return results
