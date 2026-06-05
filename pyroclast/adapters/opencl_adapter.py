"""PyOpenCL Adapter — concrete GPU implementation of the IComputeAdapter Port.

This module contains the infrastructure-facing side of the Ports & Adapters
architecture for the compute layer.  It is the **only** file in the entire
codebase that imports ``pyopencl``; all other modules depend exclusively on the
:class:`~pyroclast.ABCs.compute.IComputeAdapter` interface.

Architectural role
------------------
``PyOpenCLAdapter`` is a *secondary adapter* (driven adapter): it is called by
the Service Layer via the ``IComputeAdapter`` Port and translates the
domain-level request into low-level PyOpenCL API calls.

Device selection strategy
-------------------------
On construction the adapter scans all OpenCL platforms for a GPU device.
The **first** GPU found is used.  If no GPU is available (e.g. on a CPU-only
CI machine) the adapter falls back to ``pyopencl.create_some_context`` which
picks any available device, possibly a CPU OpenCL implementation.  This makes
the adapter usable in test environments without a dedicated GPU.

Kernel compilation
------------------
The OpenCL source is read at construction time from
``pyroclast/kernels/preprocessing.cl`` and compiled for the selected device.
Compilation errors are surfaced as ``pyopencl.RuntimeError`` with the full
compiler log.

Memory management
-----------------
* The invasion-map buffer is allocated on the device **once** per
  ``batch_preprocess`` call and reused across all habitats in the batch.
* Per-habitat input and output buffers are allocated, used, and released
  inside each loop iteration to minimise peak VRAM usage.
* All buffers are released before the method returns (both on success and on
  exception via ``try/finally``).

See also
--------
pyroclast.ABCs.compute.IComputeAdapter : the Port this class implements.
pyroclast.kernels.preprocessing.cl : the OpenCL kernel source.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.ABCs.compute import IComputeAdapter
from pyroclast.ABCs.repository import RasterMap
from pyroclast.domain.models import BenchResult, CompactedHabitat

logger = logging.getLogger(__name__)

_KERNEL_NAME = "map_multiply"


def _find_gpu_device() -> cl.Device | None:
    """Scan all OpenCL platforms and return the first GPU device found.

    Returns
    -------
    pyopencl.Device or None
        The first ``CL_DEVICE_TYPE_GPU`` device encountered across all
        platforms, or ``None`` if no GPU is available.
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

    First attempts to create a context on the first available GPU.  If no GPU
    is found, falls back to ``pyopencl.create_some_context(interactive=False)``
    which selects any available device (e.g. a CPU OpenCL implementation).

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
            "PyOpenCLAdapter: using GPU device '%s' on platform '%s'.",
            gpu.name,
            gpu.platform.name,
        )
        return cl.Context(devices=[gpu])
    logger.warning(
        "PyOpenCLAdapter: no GPU found — falling back to create_some_context()."
    )
    return cl.create_some_context(interactive=False)


class PyOpenCLAdapter(IComputeAdapter):
    """GPU compute adapter that implements the preprocessing pipeline via PyOpenCL.

    This adapter fulfils the :class:`~pyroclast.ABCs.compute.IComputeAdapter`
    contract using OpenCL kernels.  It is constructed once and can be reused
    across multiple ``batch_preprocess`` calls.

    Construction performs three one-time operations:

    1. **Device discovery** — selects the first GPU (or any device as a
       fallback).
    2. **Context and queue creation** — initialises the OpenCL runtime.
    3. **Kernel compilation** — reads ``preprocessing.cl`` and builds the
       ``map_multiply`` kernel for the selected device.

    Parameters
    ----------
    kernel_path : pathlib.Path, optional
        Absolute or relative path to the OpenCL kernel source file.
        Defaults to ``pyroclast/kernels/preprocessing.cl`` resolved relative
        to this module's location.  Override in tests to use a stub kernel.

    Raises
    ------
    FileNotFoundError
        If ``kernel_path`` does not exist or is not readable.
    pyopencl.RuntimeError
        If the OpenCL kernel fails to compile (the full compiler log is
        included in the exception message).
    pyopencl.Error
        If no OpenCL platform or device is available.

    Examples
    --------
    >>> from pyroclast.adapters.opencl_adapter import PyOpenCLAdapter
    >>> adapter = PyOpenCLAdapter()
    >>> results = adapter.batch_preprocess(invasion_map, habitats)
    """

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "preprocessing" / "preprocessing.cl"
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
        # each entry: (elapsed_ms, total_bytes)
        self._kernel_launches: list[tuple[float, int]] = []
        self._last_n_cells: int = 0
        self._last_shape: tuple[int, int] = (0, 0)

        kernel_source = kernel_path.read_text(encoding="utf-8")
        try:
            self._program: cl.Program = cl.Program(
                self._ctx, kernel_source
            ).build()
        except cl.RuntimeError as exc:
            raise RuntimeError(
                f"OpenCL kernel compilation failed.\n"
                f"Kernel path: {kernel_path}\n"
                f"Build log:\n{exc}"
            ) from exc

        self._kernel: cl.Kernel = cl.Kernel(self._program, _KERNEL_NAME)
        logger.info(
            "PyOpenCLAdapter: kernel '%s' compiled successfully.", _KERNEL_NAME
        )

    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        """Execute the Map kernel + stream compaction for a batch of habitats.

        The invasion probability raster is transferred to VRAM exactly once
        per call.  For each habitat the method:

        1. **H2D** — copies the habitat ``uint8`` raster to a read-only device
           buffer.
        2. **Kernel** — launches ``map_multiply`` with a 1-D NDRange of
           ``total_cells`` work-items, computing
           ``out[i] = p_map[i] * (float)h_map[i]`` for all ``i``.
        3. **D2H** — copies the output buffer back to host RAM.
        4. **Stream compaction** — uses ``numpy`` to select cells where
           ``out > 0``, building the ``p_vec`` array.
        5. **Cleanup** — releases the per-habitat device buffers.

        After the loop the invasion buffer is released and the method returns.

        Parameters
        ----------
        invasion_map : RasterMap
            Invasion-probability raster.  ``invasion_map.data`` must be a 2-D
            ``numpy.ndarray`` with ``dtype=float32``.
        habitats : Sequence[RasterMap]
            Habitat-presence rasters.  Each ``habitat.data`` must be a 2-D
            ``numpy.ndarray`` with ``dtype=uint8`` and the same shape as
            ``invasion_map.data``.  An empty sequence is valid.

        Returns
        -------
        list[CompactedHabitat]
            One :class:`~pyroclast.domain.models.CompactedHabitat` per
            element in ``habitats``, in the same order.

        Raises
        ------
        ValueError
            If the shapes of ``invasion_map.data`` and any ``habitat.data``
            do not match.
        pyopencl.Error
            On any OpenCL runtime error during buffer operations or kernel
            execution.
        """
        if not habitats:
            return []

        p_flat: np.ndarray = np.ascontiguousarray(
            invasion_map.data.ravel(), dtype=np.float32
        )
        total_cells = p_flat.size
        self._last_n_cells = total_cells
        self._last_shape = (invasion_map.data.shape[0], invasion_map.data.shape[1])
        mf = cl.mem_flags

        p_buf: cl.Buffer = cl.Buffer(
            self._ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=p_flat,
        )
        logger.debug(
            "PyOpenCLAdapter: invasion buffer allocated (%d cells, %.2f MiB).",
            total_cells,
            p_flat.nbytes / 1_048_576,
        )

        results: list[CompactedHabitat] = []
        try:
            for habitat in habitats:
                if habitat.data.shape != invasion_map.data.shape:
                    raise ValueError(
                        f"Habitat '{habitat.code}' shape {habitat.data.shape} "
                        f"does not match invasion map shape {invasion_map.data.shape}."
                    )

                h_flat: np.ndarray = np.ascontiguousarray(
                    habitat.data.ravel(), dtype=np.uint8
                )
                out_flat: np.ndarray = np.empty(total_cells, dtype=np.float32)

                h_buf: cl.Buffer | None = None
                out_buf: cl.Buffer | None = None
                try:
                    h_buf = cl.Buffer(
                        self._ctx,
                        mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=h_flat,
                    )
                    out_buf = cl.Buffer(
                        self._ctx,
                        mf.WRITE_ONLY,
                        size=out_flat.nbytes,
                    )

                    event = self._kernel(
                        self._queue,
                        (total_cells,),
                        None,
                        p_buf,
                        h_buf,
                        out_buf,
                        np.int32(total_cells),
                    )

                    cl.enqueue_copy(self._queue, out_flat, out_buf)
                    self._queue.finish()
                finally:
                    if h_buf is not None:
                        h_buf.release()
                    if out_buf is not None:
                        out_buf.release()

                t0_comp = time.perf_counter()
                p_vec = self._compact(out_flat)
                t_comp_ms = (time.perf_counter() - t0_comp) * 1e3
                n_cells = int(p_vec.size)

                if self._profiling:
                    elapsed_ms = (event.profile.end - event.profile.start) * 1e-6 if event is not None else 0.0
                    total_time_ms = elapsed_ms + t_comp_ms
                    total_bytes = self.get_bytes_read(invasion_map, habitat) + self.get_bytes_written(invasion_map, habitat)
                    self._kernel_launches.append((total_time_ms, total_bytes))

                results.append(
                    CompactedHabitat(
                        habitat_code=habitat.code,
                        n_cells=n_cells,
                        p_vec=p_vec,
                    )
                )
                logger.debug(
                    "PyOpenCLAdapter: habitat '%s' — %d active cells (%.1f%%).",
                    habitat.code,
                    n_cells,
                    100.0 * n_cells / total_cells if total_cells else 0.0,
                )
        finally:
            p_buf.release()

        return results

    def _compact(self, out_flat: np.ndarray) -> np.ndarray:
        """Perform host-side stream compaction using NumPy boolean masking."""
        mask = out_flat > 0.0
        return out_flat[mask].copy()

    def reset_profile(self) -> None:
        """Clear accumulated kernel timing data."""
        self._kernel_launches.clear()

    def get_bytes_read(self, invasion_map: RasterMap, habitat: RasterMap) -> int:
        """Calculate bytes read from VRAM (p_map + h_map)."""
        return invasion_map.data.nbytes + habitat.data.nbytes

    def get_bytes_written(self, invasion_map: RasterMap, habitat: RasterMap) -> int:
        """Calculate bytes written to VRAM (out_map)."""
        # out_map has the same shape and type as invasion_map.data
        return invasion_map.data.nbytes

    def benchmark(self) -> list[BenchResult]:
        """Return timing and bandwidth statistics from real kernel executions.

        Returns
        -------
        list[BenchResult]
            Timing and bandwidth statistics derived from real kernel launches.

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
                "Call batch_preprocess() at least once before benchmark()."
            )

        times_ms = [t for t, _ in self._kernel_launches]
        mean_ms = float(np.mean(times_ms))
        min_ms = float(np.min(times_ms))
        
        total_time_s = sum(times_ms) * 1e-3
        total_bytes = sum(b for _, b in self._kernel_launches)
        bandwidth_gbs = total_bytes / total_time_s / 1e9
        
        # Memory footprint in MB (Decimal: 10^6)
        memory_mb = total_bytes / len(times_ms) / 1e6
        
        return [BenchResult(
            kernel_name=_KERNEL_NAME,
            shape=self._last_shape,
            n_cells=self._last_n_cells,
            n_runs=len(times_ms),
            mean_ms=mean_ms,
            min_ms=min_ms,
            bandwidth_gbs=bandwidth_gbs,
            memory_mb=memory_mb,
        )]


class PyOpenCLHostScalarCompactionAdapter(PyOpenCLAdapter):
    """Host-side stream compaction using a pure Python scalar loop (list comprehension)."""
    
    def _compact(self, out_flat: np.ndarray) -> np.ndarray:
        return np.array([x for x in out_flat if x > 0.0], dtype=np.float32)


class PyOpenCLHostNonzeroCompactionAdapter(PyOpenCLAdapter):
    """Host-side stream compaction using NumPy flatnonzero."""
    
    def _compact(self, out_flat: np.ndarray) -> np.ndarray:
        return out_flat[np.flatnonzero(out_flat)].copy()


class PyOpenCLHostCompressCompactionAdapter(PyOpenCLAdapter):
    """Host-side stream compaction using NumPy compress."""
    
    def _compact(self, out_flat: np.ndarray) -> np.ndarray:
        return np.compress(out_flat > 0.0, out_flat)


class PyOpenCLGPUScalarCompactionAdapter(PyOpenCLAdapter):
    """GPU-side stream compaction using scalar prefix scan and compaction kernels."""
    
    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        if not habitats:
            return []

        p_flat = np.ascontiguousarray(
            invasion_map.data.ravel(), dtype=np.float32
        )
        total_cells = p_flat.size
        self._last_n_cells = total_cells
        self._last_shape = (invasion_map.data.shape[0], invasion_map.data.shape[1])
        mf = cl.mem_flags

        p_buf = cl.Buffer(
            self._ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=p_flat,
        )

        results = []
        try:
            k_map = cl.Kernel(self._program, "map_multiply")
            k_gen_pred = cl.Kernel(self._program, "generate_predicates")
            k_scan = cl.Kernel(self._program, "scan_scalar_k")
            k_corr = cl.Kernel(self._program, "scan_correction_scalar_k")
            k_compact = cl.Kernel(self._program, "stream_compaction_scalar_k")

            for habitat in habitats:
                if habitat.data.shape != invasion_map.data.shape:
                    raise ValueError(
                        f"Habitat '{habitat.code}' shape {habitat.data.shape} "
                        f"does not match invasion map shape {invasion_map.data.shape}."
                    )

                h_flat = np.ascontiguousarray(
                    habitat.data.ravel(), dtype=np.uint8
                )

                h_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_flat)
                out_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=p_flat.nbytes)
                predicates_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=total_cells * 4)
                scanned_predicates_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=total_cells * 4)

                LWS = 256
                nwg = min(64, (total_cells + LWS - 1) // LWS)
                code_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=nwg * 4)

                # 1. Run map_multiply
                evt_map = k_map(self._queue, (total_cells,), None, p_buf, h_buf, out_buf, np.int32(total_cells))
                
                # 2. Run generate_predicates
                evt_gen = k_gen_pred(self._queue, (total_cells,), None, h_buf, predicates_buf, np.int32(total_cells))

                # 3. Run scan_scalar_k
                lmem = cl.LocalMemory(LWS * 4)
                evt_scan = k_scan(
                    self._queue,
                    (nwg * LWS,),
                    (LWS,),
                    scanned_predicates_buf,
                    code_buf,
                    predicates_buf,
                    lmem,
                    np.int32(total_cells)
                )

                # 4. Correct group-level sums if nwg > 1
                evt_corr = None
                code_scanned_buf = None
                code_host = np.empty(nwg, dtype=np.int32)
                cl.enqueue_copy(self._queue, code_host, code_buf)
                self._queue.finish()

                if nwg > 1:
                    code_scanned = np.cumsum(code_host).astype(np.int32)
                    n_cells = int(code_scanned[-1])

                    code_scanned_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=code_scanned)
                    evt_corr = k_corr(
                        self._queue,
                        (nwg * LWS,),
                        (LWS,),
                        scanned_predicates_buf,
                        code_scanned_buf,
                        np.int32(total_cells)
                    )
                else:
                    n_cells = int(code_host[0])

                if n_cells == 0:
                    p_vec = np.array([], dtype=np.float32)
                    evt_comp = None
                else:
                    compacted_p_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, size=n_cells * 4)
                    
                    # 5. Run stream_compaction_scalar_k
                    comp_gws = ((total_cells + LWS - 1) // LWS) * LWS
                    evt_comp = k_compact(
                        self._queue,
                        (comp_gws,),
                        (LWS,),
                        out_buf,
                        scanned_predicates_buf,
                        predicates_buf,
                        compacted_p_buf,
                        np.int32(total_cells)
                    )
                    
                    p_vec = np.empty(n_cells, dtype=np.float32)
                    cl.enqueue_copy(self._queue, p_vec, compacted_p_buf)
                    self._queue.finish()
                    compacted_p_buf.release()

                # Clean up per-habitat buffers
                h_buf.release()
                out_buf.release()
                predicates_buf.release()
                scanned_predicates_buf.release()
                code_buf.release()
                if code_scanned_buf is not None:
                    code_scanned_buf.release()

                results.append(
                    CompactedHabitat(
                        habitat_code=habitat.code,
                        n_cells=n_cells,
                        p_vec=p_vec,
                    )
                )

                if self._profiling:
                    total_evt_time = sum(
                        (evt.profile.end - evt.profile.start) * 1e-6
                        for evt in [evt_map, evt_gen, evt_scan, evt_corr, evt_comp]
                        if evt is not None
                    )
                    total_bytes = self.get_bytes_read(invasion_map, habitat) + (n_cells * 4)
                    self._kernel_launches.append((total_evt_time, total_bytes))
        finally:
            p_buf.release()

        return results


class PyOpenCLGPUVectorizedCompactionAdapter(PyOpenCLAdapter):
    """GPU-side stream compaction using vectorized (int4) prefix scan and compaction kernels."""
    
    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        if not habitats:
            return []

        p_flat = np.ascontiguousarray(
            invasion_map.data.ravel(), dtype=np.float32
        )
        total_cells = p_flat.size
        nquads = (total_cells + 3) // 4
        self._last_n_cells = total_cells
        self._last_shape = (invasion_map.data.shape[0], invasion_map.data.shape[1])
        mf = cl.mem_flags

        p_buf = cl.Buffer(
            self._ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=p_flat,
        )

        results = []
        try:
            k_map = cl.Kernel(self._program, "map_multiply")
            k_gen_pred = cl.Kernel(self._program, "generate_predicates")
            k_scan = cl.Kernel(self._program, "scan_k")
            k_corr = cl.Kernel(self._program, "scan_correction_k")
            k_compact = cl.Kernel(self._program, "stream_compaction_k")

            for habitat in habitats:
                if habitat.data.shape != invasion_map.data.shape:
                    raise ValueError(
                        f"Habitat '{habitat.code}' shape {habitat.data.shape} "
                        f"does not match invasion map shape {invasion_map.data.shape}."
                    )

                h_flat = np.ascontiguousarray(
                    habitat.data.ravel(), dtype=np.uint8
                )

                h_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_flat)
                out_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=p_flat.nbytes)
                
                predicates_host = np.zeros(nquads * 4, dtype=np.int32)
                predicates_buf = cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=predicates_host)
                scanned_predicates_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=nquads * 16)

                LWS = 256
                nwg = min(64, (nquads + LWS - 1) // LWS)
                code_buf = cl.Buffer(self._ctx, mf.READ_WRITE, size=nwg * 4)

                # 1. Run map_multiply
                evt_map = k_map(self._queue, (total_cells,), None, p_buf, h_buf, out_buf, np.int32(total_cells))
                
                # 2. Run generate_predicates
                evt_gen = k_gen_pred(self._queue, (total_cells,), None, h_buf, predicates_buf, np.int32(total_cells))

                # 3. Run scan_k
                lmem = cl.LocalMemory(LWS * 4)
                evt_scan = k_scan(
                    self._queue,
                    (nwg * LWS,),
                    (LWS,),
                    scanned_predicates_buf,
                    code_buf,
                    predicates_buf,
                    lmem,
                    np.int32(nquads)
                )

                # 4. Correct group-level sums if nwg > 1
                evt_corr = None
                code_scanned_buf = None
                code_host = np.empty(nwg, dtype=np.int32)
                cl.enqueue_copy(self._queue, code_host, code_buf)
                self._queue.finish()

                if nwg > 1:
                    code_scanned = np.cumsum(code_host).astype(np.int32)
                    n_cells = int(code_scanned[-1])

                    code_scanned_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=code_scanned)
                    evt_corr = k_corr(
                        self._queue,
                        (nwg * LWS,),
                        (LWS,),
                        scanned_predicates_buf,
                        code_scanned_buf,
                        np.int32(nquads)
                    )
                else:
                    n_cells = int(code_host[0])

                if n_cells == 0:
                    p_vec = np.array([], dtype=np.float32)
                    evt_comp = None
                else:
                    compacted_p_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, size=n_cells * 4)
                    
                    # 5. Run stream_compaction_k
                    comp_gws = ((nquads + LWS - 1) // LWS) * LWS
                    evt_comp = k_compact(
                        self._queue,
                        (comp_gws,),
                        (LWS,),
                        out_buf,
                        scanned_predicates_buf,
                        predicates_buf,
                        compacted_p_buf,
                        np.int32(total_cells),
                        np.int32(nquads)
                    )
                    
                    p_vec = np.empty(n_cells, dtype=np.float32)
                    cl.enqueue_copy(self._queue, p_vec, compacted_p_buf)
                    self._queue.finish()
                    compacted_p_buf.release()

                # Clean up per-habitat buffers
                h_buf.release()
                out_buf.release()
                predicates_buf.release()
                scanned_predicates_buf.release()
                code_buf.release()
                if code_scanned_buf is not None:
                    code_scanned_buf.release()

                results.append(
                    CompactedHabitat(
                        habitat_code=habitat.code,
                        n_cells=n_cells,
                        p_vec=p_vec,
                    )
                )

                if self._profiling:
                    total_evt_time = sum(
                        (evt.profile.end - evt.profile.start) * 1e-6
                        for evt in [evt_map, evt_gen, evt_scan, evt_corr, evt_comp]
                        if evt is not None
                    )
                    total_bytes = self.get_bytes_read(invasion_map, habitat) + (n_cells * 4)
                    self._kernel_launches.append((total_evt_time, total_bytes))
        finally:
            p_buf.release()

        return results
