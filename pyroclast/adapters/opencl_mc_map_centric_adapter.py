"""PyOpenCL Map-Centric Monte Carlo Adapter.

Adapter for the ``monte_carlo_map_centric.cl`` kernel: a 1-D NDRange kernel
that sweeps the *whole* geographic map once and, via a 64-bit presence bitmask,
updates the invaded-cell count of every habitat occupying a cell from a single
RNG draw. Overlapping habitats therefore cost no extra random numbers — the
key advantage over the Habitat-Centric kernels, which re-sample shared cells
once per habitat.

Architectural role
------------------
A *secondary adapter* (driven adapter). It reuses the OpenCL plumbing of
:class:`~pyroclast.adapters.opencl_mc_adapter.PyOpenCLMonteCarloAdapter`
(device discovery, context/queue, kernel compilation with the ``misc.h`` and
``mwc64x`` include paths) but exposes a *multi-habitat* entry point,
:meth:`run_map`, instead of the single-habitat ``run`` / ``run_batched`` API.

Batching
--------
At most ``MAX_BATCH_SIZE`` (= 64) habitats fit in one ``ulong`` bitmask, so the
host splits the habitat list into chunks of 64, launches the kernel once per
chunk, reduces the per-habitat partials on the host, and recomposes a single
``{habitat_code: probability}`` mapping.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyopencl as cl  # type: ignore[import-untyped]

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig, SpatialHabitat

logger = logging.getLogger(__name__)

# Bits available in one ulong presence mask → habitats processed per launch.
MAX_BATCH_SIZE = 64


def _floor_pow2(n: int) -> int:
    """Return the largest power of two that is ``<= n`` (and ``>= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


class PyOpenCLMapCentricAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter for the Map-Centric, 64-bit-bitmask Monte Carlo sampling kernel.

    Construction reuses :class:`PyOpenCLMonteCarloAdapter.__init__` (device
    discovery, context/queue, kernel compilation). Only the sampling kernel
    differs, selected via ``_SAMPLING_KERNEL_NAME``.
    """

    _SAMPLING_KERNEL_NAME = "monte_carlo_map_centric"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo"
                / "monte_carlo_map_centric.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)

    # ------------------------------------------------------------------
    # Single-habitat API is not applicable to the Map-Centric kernel.
    # ------------------------------------------------------------------

    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        raise NotImplementedError(
            "PyOpenCLMapCentricAdapter is multi-habitat; use run_map()."
        )

    def run_batched(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
        n_batches: int,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> float:
        raise NotImplementedError(
            "PyOpenCLMapCentricAdapter is multi-habitat; use run_map()."
        )

    # ------------------------------------------------------------------
    # Map-Centric API
    # ------------------------------------------------------------------

    def _choose_lws(self, num_habitats: int) -> int:
        """Largest power-of-two LWS whose 2-D scratch fits in local memory.

        The kernel reduces ``private_destroyed`` over a local scratch of
        ``lws * num_habitats`` ints, so ``lws * num_habitats * 4`` bytes must
        fit in ``CL_DEVICE_LOCAL_MEM_SIZE``. A power-of-two LWS is also required
        for the work-group tree reduction to be exact.
        """
        device = self._ctx.devices[0]
        max_wg = _floor_pow2(int(device.max_work_group_size))
        local_mem = int(device.local_mem_size)
        lws = min(256, max_wg)
        while lws > 1 and lws * num_habitats * 4 > local_mem:
            lws //= 2
        return lws

    def _run_chunk(
        self,
        p_flat: np.ndarray,
        chunk: list[SpatialHabitat],
        config: MonteCarloConfig,
    ) -> np.ndarray:
        """Launch the kernel for one chunk of <= 64 habitats.

        Returns the per-habitat destruction *counts* (int64, length
        ``len(chunk)``) summed across all work-groups.
        """
        n_map_cells = int(p_flat.size)
        num_habitats = len(chunk)

        # Build the per-cell presence bitmask and the per-habitat metadata.
        habitat_mask = np.zeros(n_map_cells, dtype=np.uint64)
        hab_total_cells = np.zeros(num_habitats, dtype=np.uint32)
        hab_thresholds = np.zeros(num_habitats, dtype=np.float32)
        for h, hab in enumerate(chunk):
            idx = np.flatnonzero(hab.presence_mask.ravel())
            habitat_mask[idx] |= np.uint64(1) << np.uint64(h)
            hab_total_cells[h] = idx.size
            hab_thresholds[h] = hab.threshold

        # Topology: power-of-two LWS sized so the 2-D scratch fits; enough
        # work-groups to saturate the device.
        lws = self._choose_lws(num_habitats)
        max_cu = int(self._ctx.devices[0].max_compute_units)
        n_wg = max_cu * 4
        gws = n_wg * lws

        mf = cl.mem_flags
        p_buf: cl.Buffer | None = None
        mask_buf: cl.Buffer | None = None
        total_buf: cl.Buffer | None = None
        thr_buf: cl.Buffer | None = None
        partial_buf: cl.Buffer | None = None
        try:
            p_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_flat
            )
            mask_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=habitat_mask
            )
            total_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hab_total_cells
            )
            thr_buf = cl.Buffer(
                self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hab_thresholds
            )
            partial_buf = cl.Buffer(
                self._ctx, mf.READ_WRITE, size=num_habitats * n_wg * 4
            )

            event = self._kernel(
                self._queue,
                (gws,),
                (lws,),
                p_buf,
                mask_buf,
                total_buf,
                thr_buf,
                partial_buf,
                cl.LocalMemory(4 * lws * num_habitats),
                np.uint32(n_map_cells),
                np.uint32(num_habitats),
                np.uint64(int(config.seed)),
                np.uint32(config.n_runs),
            )

            if self._profiling:
                event.wait()
                elapsed_ms = (event.profile.end - event.profile.start) * 1e-6
                self._last_n_cells = n_map_cells
                self._last_n_wg = n_wg
                self._last_kernel_name = self._SAMPLING_KERNEL_NAME
                self._kernel_launches.append((elapsed_ms, 0))

            # Host-side reduction of the small [num_habitats x n_wg] partials.
            partial_host = np.empty(num_habitats * n_wg, dtype=np.int32)
            cl.enqueue_copy(self._queue, partial_host, partial_buf)
            self._queue.finish()
        finally:
            for buf in (p_buf, mask_buf, total_buf, thr_buf, partial_buf):
                if buf is not None:
                    buf.release()

        return partial_host.reshape(num_habitats, n_wg).sum(axis=1, dtype=np.int64)

    def run_map(
        self,
        p_map: np.ndarray,
        habitats: list[SpatialHabitat],
        config: MonteCarloConfig,
    ) -> dict[str, float]:
        """Estimate destruction probability for every habitat in one map sweep.

        Parameters
        ----------
        p_map : numpy.ndarray
            2-D ``float32`` invasion-probability map. Every habitat's
            ``presence_mask`` must share this shape.
        habitats : list[SpatialHabitat]
            Habitats to evaluate. Processed in chunks of ``MAX_BATCH_SIZE``.
        config : MonteCarloConfig
            Carries ``n_runs`` and ``seed``. Per-habitat thresholds come from
            each :class:`SpatialHabitat`; ``config.threshold`` is unused here.

        Returns
        -------
        dict[str, float]
            ``{habitat_code: P(destruction)}`` in the input order. Empty
            habitats (no occupied cells) map to ``0.0``.
        """
        if p_map.ndim != 2:
            raise ValueError(
                f"p_map must be 2-D, got shape {p_map.shape}."
            )
        for hab in habitats:
            if hab.presence_mask.shape != p_map.shape:
                raise ValueError(
                    f"habitat '{hab.habitat_code}' mask shape "
                    f"{hab.presence_mask.shape} != map shape {p_map.shape}."
                )

        p_flat = np.ascontiguousarray(p_map.ravel(), dtype=np.float32)

        # Empty habitats can never be destroyed; assign 0.0 and skip the kernel
        # (also keeps hab_total_cells strictly positive, avoiding a div-by-0).
        results: dict[str, float] = {}
        non_empty = [h for h in habitats if h.n_cells > 0]
        for hab in habitats:
            if hab.n_cells == 0:
                results[hab.habitat_code] = 0.0

        for start in range(0, len(non_empty), MAX_BATCH_SIZE):
            chunk = non_empty[start : start + MAX_BATCH_SIZE]
            counts = self._run_chunk(p_flat, chunk, config)
            for hab, count in zip(chunk, counts):
                results[hab.habitat_code] = int(count) / config.n_runs

        logger.debug(
            "PyOpenCLMapCentricAdapter.run_map: %d habitats (%d non-empty) over "
            "%d-cell map — R=%d, seed=%d.",
            len(habitats),
            len(non_empty),
            p_flat.size,
            config.n_runs,
            config.seed,
        )
        # Recompose in input order.
        return {hab.habitat_code: results[hab.habitat_code] for hab in habitats}
