"""PyOpenCL 2-D Grid-Stride Monte Carlo Adapter.

Adapter for the ``monte_carlo_2d_stride.cl`` kernel: 2-D NDRange with
grid-stride sampling and an in-place sequential-addressing tree reduction
over a single linearised local scratch buffer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, GridTopology, MonteCarloConfig

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarlo2DAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter for the 2D grid-stride Monte Carlo sampling kernel."""

    _SAMPLING_KERNEL_NAME = "monte_carlo_2d_stride"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo_2d_stride.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)

    def suggest_topology(self, n_runs: int) -> GridTopology:
        """Suggest a 2D execution grid optimised for the device."""
        device = self._ctx.devices[0]
        max_cu = device.max_compute_units

        lws = (32, 8)
        n_wgs_x = max_cu * 4
        n_wgs_y = 2

        gws = (n_wgs_x * lws[0], n_wgs_y * lws[1])
        return GridTopology(gws=gws, lws=lws)

    @staticmethod
    def _get_sizes(
        topology: GridTopology,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Return (gws_tuple, lws_tuple, wg_size) regardless of 1-D/2-D."""
        gws = topology.gws
        lws = topology.lws

        if isinstance(gws, int):
            return (gws,), (lws,), int(lws)  # type: ignore[arg-type]

        gws_arg = tuple(int(x) for x in gws)
        lws_arg = tuple(int(x) for x in lws)  # type: ignore[arg-type]
        wg_size = 1
        for dim in lws_arg:
            wg_size *= dim
        return gws_arg, lws_arg, wg_size

    def _allocate_partial_buffers(
        self, n_wg: int
    ) -> tuple[cl.Buffer, cl.Buffer]:
        mf = cl.mem_flags
        size = max(1, n_wg) * 4
        return (
            cl.Buffer(self._ctx, mf.READ_WRITE, size=size),
            cl.Buffer(self._ctx, mf.READ_WRITE, size=size),
        )

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Execute the 2D Monte Carlo kernel followed by the recursive reduce."""
        p_host = np.ascontiguousarray(habitat.p_vec, dtype=np.float32)

        topology = config.topology or self.suggest_topology(config.n_runs)
        gws_arg, lws_arg, wg_size = self._get_sizes(topology)
        total_threads = 1
        for g in gws_arg:
            total_threads *= g
        n_wg = total_threads // wg_size

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
            partial_a, partial_b = self._allocate_partial_buffers(n_wg)

            event = self._kernel(
                self._queue,
                gws_arg,
                lws_arg,
                p_buf,
                partial_a,
                np.uint32(habitat.n_cells),
                np.float32(config.threshold),
                np.uint64(int(config.seed)),
                np.uint32(config.n_runs),
                cl.LocalMemory(4 * wg_size),
            )

            if self._profiling:
                event.wait()
                elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                self._last_n_cells = habitat.n_cells
                self._last_n_wg = n_wg
                total_bytes = self.get_bytes_read(
                    habitat, config.n_runs
                ) + self.get_bytes_written(habitat, config.n_runs)
                self._kernel_launches.append((elapsed_ms, total_bytes))

            total_count = self._reduce_partial(partial_a, partial_b, n_wg)
        finally:
            for buf in (p_buf, partial_a, partial_b):
                if buf is not None:
                    buf.release()

        return total_count / config.n_runs

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
        gws_arg, lws_arg, wg_size = self._get_sizes(topology)
        total_threads = 1
        for g in gws_arg:
            total_threads *= g
        n_wg = total_threads // wg_size

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
            partial_a, partial_b = self._allocate_partial_buffers(n_wg)

            total_count = 0
            for i in range(n_batches):
                event = self._kernel(
                    self._queue,
                    gws_arg,
                    lws_arg,
                    p_buf,
                    partial_a,
                    np.uint32(habitat.n_cells),
                    np.float32(config.threshold),
                    np.uint64(int(config.seed) + i * batch_size * habitat.n_cells),
                    np.uint32(batch_size),
                    cl.LocalMemory(4 * wg_size),
                )
                if self._profiling:
                    event.wait()
                    elapsed_ms = (
                        event.profile.end - event.profile.start
                    ) * 1e-6
                    self._last_n_cells = habitat.n_cells
                    self._last_n_wg = n_wg
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

        return total_count / config.n_runs
