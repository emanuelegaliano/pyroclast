"""PyOpenCL 2-D Monte Carlo Adapter — transposed launch grid (runs in X).

Same algorithm and results as :class:`PyOpenCLMonteCarlo2DAdapter`, but the grid
axes are swapped: the run axis is on dim0 (the hardware-fast axis) and the cell
axis is on dim1. This drives ``pyroclast/kernels/monte_carlo_2d_transposed.cl``
and exists to measure the effect of the launch-grid mapping (p_vec read pattern
and local-memory bank conflicts) against the non-transposed kernel.

Only the three topology hooks and the kernel path/name differ from the base.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_2d_adapter import PyOpenCLMonteCarlo2DAdapter

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarlo2DTransposedAdapter(PyOpenCLMonteCarlo2DAdapter):
    """2-D Monte Carlo adapter with the run axis on dim0 and the cell axis on dim1."""

    _SAMPLING_KERNEL_NAME = "monte_carlo_2d_transposed"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        cell_lanes: int = 64,
        run_lanes: int = 4,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo"
                / "monte_carlo_2d_transposed.cl"
            )
        super().__init__(
            kernel_path=kernel_path,
            profiling=profiling,
            cell_lanes=cell_lanes,
            run_lanes=run_lanes,
        )

    def _build_topology(
        self, cell_lanes: int, run_lanes: int, n_wg_runs: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Transposed layout: runs on dim0, cells on dim1."""
        gws = (n_wg_runs * run_lanes, cell_lanes)
        lws = (run_lanes, cell_lanes)
        return gws, lws

    def _run_axis_groups(
        self, gws: tuple[int, int], lws: tuple[int, int]
    ) -> int:
        """Number of work-groups along the run axis (dim0)."""
        return gws[0] // lws[0]

    def _cell_run_lanes_from_lws(self, lws: tuple[int, int]) -> tuple[int, int]:
        """Transposed lws is (run_lanes, cell_lanes); return (cell_lanes, run_lanes)."""
        return lws[1], lws[0]
