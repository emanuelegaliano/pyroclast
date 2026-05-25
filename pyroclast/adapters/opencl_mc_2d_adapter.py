"""PyOpenCL 2-D Monte Carlo Adapter — array in X, runs in Y.

This adapter drives ``pyroclast/kernels/monte_carlo_2d.cl``, which parallelises
the per-run cell scan (the serial ``_count_invaded`` of the 1-D kernel) across a
second grid axis of ``cell_lanes`` (Lc) threads and closes each run with two
in-kernel reductions: first over the cell lanes, then over the ``run_lanes`` (Rr)
of a work-group. The host then sums the per-work-group partials with the shared
``reduce_sum_int`` reducer, exactly like the 1-D adapters.

The cell axis uses the *interleaved* partition of ``misc_2d.h`` (cell lane L
handles cells L, L+Lc, ...), so consecutive lanes read consecutive ``p_vec``
addresses (coalesced). That layout matches the vectorized kernel's stream with
``VEC_WIDTH == Lc``: independence is preserved but it is **not** bit-exact with
the scalar 1-D kernel, so it is validated statistically (and is bit-exact with
``PyOpenCLMonteCarloVectorizedAdapter`` of equal width). ``p_vec`` is padded to
the per-run stride with a ``-1.0`` sentinel, as in the vectorized adapter.

Subclasses select the transposed launch grid by overriding the three small
topology hooks (``_build_topology``, ``_run_axis_groups``,
``_cell_run_lanes_from_lws``) and the kernel path/name.
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


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class PyOpenCLMonteCarlo2DAdapter(PyOpenCLMonteCarloAdapter):
    """2-D Monte Carlo adapter with the cell axis on dim0 and the run axis on dim1."""

    _SAMPLING_KERNEL_NAME = "monte_carlo_2d"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        cell_lanes: int = 64,
        run_lanes: int = 4,
    ) -> None:
        if not _is_pow2(cell_lanes):
            raise ValueError(
                f"cell_lanes must be a power of two, got {cell_lanes}."
            )
        if not _is_pow2(run_lanes):
            raise ValueError(
                f"run_lanes must be a power of two, got {run_lanes}."
            )
        self._cell_lanes = cell_lanes
        self._run_lanes = run_lanes
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo_2d.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)
        logger.info(
            "%s initialized (cell_lanes=%d, run_lanes=%d, kernel=%s).",
            type(self).__name__,
            cell_lanes,
            run_lanes,
            kernel_path.name,
        )

    # ------------------------------------------------------------------
    # Padding (shared stream layout with the vectorized kernel)
    # ------------------------------------------------------------------

    def _run_stride(self, n_cells: int) -> int:
        """Stream positions consumed by one run = ceil(n_cells/Lc) * Lc."""
        lc = self._cell_lanes
        return ((n_cells + lc - 1) // lc) * lc

    def _padded_p_host(self, habitat: CompactedHabitat) -> np.ndarray:
        """Pad p_vec to the per-run stride with -1.0 (sentinel: never invaded)."""
        run_stride = self._run_stride(habitat.n_cells)
        p_host = np.full(run_stride, -1.0, dtype=np.float32)
        p_host[: habitat.n_cells] = np.asarray(habitat.p_vec, dtype=np.float32)
        return np.ascontiguousarray(p_host, dtype=np.float32)

    # ------------------------------------------------------------------
    # Topology hooks (overridden by the transposed variant)
    # ------------------------------------------------------------------

    def _build_topology(
        self, cell_lanes: int, run_lanes: int, n_wg_runs: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return (gws, lws) tuples. Natural layout: cells on dim0, runs on dim1."""
        gws = (cell_lanes, n_wg_runs * run_lanes)
        lws = (cell_lanes, run_lanes)
        return gws, lws

    def _run_axis_groups(
        self, gws: tuple[int, int], lws: tuple[int, int]
    ) -> int:
        """Number of work-groups along the run axis (= number of partials)."""
        return gws[1] // lws[1]

    def _cell_run_lanes_from_lws(self, lws: tuple[int, int]) -> tuple[int, int]:
        """Extract (cell_lanes, run_lanes) from an lws tuple. Natural: (Lc, Rr)."""
        return lws[0], lws[1]

    def suggest_topology(self, n_runs: int) -> GridTopology:
        """Suggest a 2-D execution grid that saturates the device."""
        device = self._ctx.devices[0]
        n_wg_runs = int(device.max_compute_units) * 4
        gws, lws = self._build_topology(
            self._cell_lanes, self._run_lanes, n_wg_runs
        )
        return GridTopology(gws=gws, lws=lws)

    def _resolve_topology(
        self, config: MonteCarloConfig
    ) -> tuple[tuple[int, int], tuple[int, int], int, int, int]:
        """Return (gws, lws, cell_lanes, run_lanes, n_wg) for this config."""
        topology = config.topology or self.suggest_topology(config.n_runs)
        gws, lws = topology.gws, topology.lws
        if not (isinstance(gws, tuple) and isinstance(lws, tuple)):
            raise ValueError(
                f"{type(self).__name__} requires a 2-D GridTopology "
                f"(tuple gws/lws); got gws={gws!r}, lws={lws!r}."
            )
        cell_lanes, run_lanes = self._cell_run_lanes_from_lws(lws)
        n_wg = self._run_axis_groups(gws, lws)
        return gws, lws, cell_lanes, run_lanes, n_wg

    # ------------------------------------------------------------------
    # IMonteCarloAdapter API
    # ------------------------------------------------------------------

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate destruction probability for a single habitat via GPU."""
        p_host = self._padded_p_host(habitat)
        gws, lws, cell_lanes, run_lanes, n_wg = self._resolve_topology(config)

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        partial_a: cl.Buffer | None = None
        partial_b: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            partial_a = cl.Buffer(self._ctx, mf.READ_WRITE, size=n_wg * 4)
            partial_b = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=max(1, n_wg) * 4
            )
            local_mem = cl.LocalMemory(4 * (cell_lanes * run_lanes + run_lanes))

            event = self._kernel(
                self._queue,
                gws,
                lws,
                p_buf,
                partial_a,
                local_mem,
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
            "%s: habitat '%s' — prob=%.4f (R=%d, N_c=%d, Lc=%d, Rr=%d).",
            type(self).__name__,
            habitat.habitat_code,
            prob,
            config.n_runs,
            habitat.n_cells,
            cell_lanes,
            run_lanes,
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
        p_host = self._padded_p_host(habitat)
        # Each run consumes run_stride stream positions, so batches step
        # base_offset by batch_size * run_stride to stay non-overlapping.
        run_stride = self._run_stride(habitat.n_cells)
        gws, lws, cell_lanes, run_lanes, n_wg = self._resolve_topology(config)

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        partial_a: cl.Buffer | None = None
        partial_b: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            partial_a = cl.Buffer(self._ctx, mf.READ_WRITE, size=n_wg * 4)
            partial_b = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=max(1, n_wg) * 4
            )
            local_mem = cl.LocalMemory(4 * (cell_lanes * run_lanes + run_lanes))

            total_count = 0
            for i in range(n_batches):
                event = self._kernel(
                    self._queue,
                    gws,
                    lws,
                    p_buf,
                    partial_a,
                    local_mem,
                    np.uint32(habitat.n_cells),
                    np.float32(config.threshold),
                    np.uint64(int(config.seed) + i * batch_size * run_stride),
                    np.uint32(batch_size),
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

    def get_bytes_read(self, habitat: CompactedHabitat, n_runs: int) -> int:
        """Bytes read by the sampling kernel: n_runs * run_stride float32 loads."""
        return n_runs * self._run_stride(habitat.n_cells) * 4
