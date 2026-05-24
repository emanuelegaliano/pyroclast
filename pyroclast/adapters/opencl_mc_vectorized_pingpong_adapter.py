"""PyOpenCL Vectorized Ping-Pong 1-D Monte Carlo Adapter.

Adapter for the ``monte_carlo_vectorized_pingpong.cl`` kernel: the same
vectorized-RNG sampling as :class:`PyOpenCLMonteCarloVectorizedAdapter`, but
with the ping-pong work-group reduction (two alternating local buffers, one
barrier per step) instead of the in-place tree reduction.

Only the kernel file differs from the vectorized base adapter; the sampling,
p_vec padding (-1.0 sentinel) and run-stride batching are inherited unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_vectorized_adapter import (
    PyOpenCLMonteCarloVectorizedAdapter,
)

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarloVectorizedPingPongAdapter(
    PyOpenCLMonteCarloVectorizedAdapter
):
    """Vectorized-RNG sampling kernel with a ping-pong work-group reduction."""

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        vec_width: int = 4,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo_vectorized_pingpong.cl"
            )
        super().__init__(
            kernel_path=kernel_path,
            profiling=profiling,
            vec_width=vec_width,
        )
