"""PyOpenCL Monte Carlo Ping-Pong Adapter — optimized GPU reduction.

This adapter uses the ``monte_carlo_pingpong.cl`` kernel, which implements
a tree reduction using two local memory arrays (ping-pong) to potentially
improve performance by avoiding some shared memory conflicts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)

class PyOpenCLMonteCarloPingPongAdapter(PyOpenCLMonteCarloAdapter):
    """GPU Monte Carlo adapter using an optimized ping-pong reduction kernel.

    Inherits from ``PyOpenCLMonteCarloAdapter`` but defaults to the
    ``monte_carlo_pingpong.cl`` kernel.
    """

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo_pingpong.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)
        logger.info(
            "PyOpenCLMonteCarloPingPongAdapter initialized with kernel: %s",
            kernel_path.name,
        )
