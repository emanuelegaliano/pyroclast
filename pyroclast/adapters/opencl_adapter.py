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
                Path(__file__).parent.parent / "kernels" / "preprocessing.cl"
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
        self._kernel_times_ms: list[float] = []
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

                    if self._profiling:
                        elapsed_ms = (
                            event.profile.end - event.profile.start
                        ) * 1e-6
                        self._kernel_times_ms.append(elapsed_ms)
                finally:
                    if h_buf is not None:
                        h_buf.release()
                    if out_buf is not None:
                        out_buf.release()

                mask: np.ndarray = out_flat > 0.0
                p_vec: np.ndarray = out_flat[mask].copy()
                n_cells = int(p_vec.size)

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

    def reset_profile(self) -> None:
        """Clear accumulated kernel timing data."""
        self._kernel_times_ms.clear()

    def benchmark(self) -> BenchResult:
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
        if not self._kernel_times_ms:
            raise ValueError(
                "No kernel executions recorded yet. "
                "Call batch_preprocess() at least once before benchmark()."
            )
        # reads: 4 B (float32 p_map) + 1 B (uint8 h_map); writes: 4 B (float32 out)
        _BYTES_PER_CELL = 9
        times = self._kernel_times_ms
        mean_ms = float(np.mean(times))
        min_ms = float(np.min(times))
        # n_cells is the same for every launch (all habitats share the invasion map shape)
        # infer it from the last recorded run via bandwidth formula inversion is not possible,
        # so we store it alongside the times
        n_cells = self._last_n_cells
        bandwidth_gbs = (_BYTES_PER_CELL * n_cells) / (mean_ms * 1e-3) / 1e9
        return BenchResult(
            kernel_name=_KERNEL_NAME,
            shape=self._last_shape,
            n_cells=n_cells,
            n_runs=len(times),
            mean_ms=mean_ms,
            min_ms=min_ms,
            bandwidth_gbs=bandwidth_gbs,
        )
