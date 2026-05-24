"""PyOpenCL Vectorized 1-D Monte Carlo Adapter.

Adapter for the ``monte_carlo_vectorized.cl`` kernel: a 1-D NDRange kernel
that uses MWC64X's vector RNG (``mwc64xvec2/4/8``) to advance ``vec_width``
independent lanes per ``Step()``. The kernel keeps the launch-independent,
position-based seeding of the scalar variant by building its vector state
from ``vec_width`` scalar seeds, so each run still owns a contiguous,
non-overlapping stream block of length ``ceil(n_cells/vec_width)*vec_width``.

Unlike the other variants this kernel is **not** bit-exact with the scalar
stream (the lane layout differs); it is validated statistically.

The host pads ``p_vec`` up to the per-run stride with ``-1.0`` so the kernel's
vector loads never run past the buffer and padded lanes (probability < 0) are
deterministically never invaded — no tail loop or masking needed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig

logger = logging.getLogger(__name__)

_VALID_WIDTHS = (2, 4, 8)


class PyOpenCLMonteCarloVectorizedAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter for the vectorized-RNG 1-D Monte Carlo sampling kernel."""

    _SAMPLING_KERNEL_NAME = "monte_carlo_vectorized"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        vec_width: int = 4,
    ) -> None:
        if vec_width not in _VALID_WIDTHS:
            raise ValueError(
                f"vec_width must be one of {_VALID_WIDTHS}, got {vec_width}."
            )
        self._vec_width = vec_width
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo_vectorized.cl"
            )
        super().__init__(
            kernel_path=kernel_path,
            profiling=profiling,
            extra_build_options=f"-DVEC_WIDTH={vec_width}",
        )

    def _padded_p_host(self, habitat: CompactedHabitat) -> np.ndarray:
        """Pad p_vec to the per-run stride (multiple of vec_width) with -1.0.

        -1.0 is a sentinel: a draw x in [0, 1) is never <= -1.0, so padded
        lanes never count as invaded regardless of the RNG output.
        """
        run_stride = self._run_stride(habitat.n_cells)
        p_host = np.full(run_stride, -1.0, dtype=np.float32)
        p_host[: habitat.n_cells] = np.asarray(habitat.p_vec, dtype=np.float32)
        return np.ascontiguousarray(p_host, dtype=np.float32)

    def _run_stride(self, n_cells: int) -> int:
        """Stream positions consumed by one run = ceil(n_cells/W) * W."""
        w = self._vec_width
        return ((n_cells + w - 1) // w) * w

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate destruction probability for a single habitat via GPU."""
        p_host = self._padded_p_host(habitat)

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
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            partial_a = cl.Buffer(self._ctx, mf.READ_WRITE, size=n_wg * 4)
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
        p_host = self._padded_p_host(habitat)
        # Each run consumes run_stride stream positions, so batches must step
        # base_offset by batch_size * run_stride to stay non-overlapping.
        run_stride = self._run_stride(habitat.n_cells)

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
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_host
            )
            partial_a = cl.Buffer(self._ctx, mf.READ_WRITE, size=n_wg * 4)
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
