"""PyOpenCL Monte Carlo Ping-Pong Adapter.

Adapter for the ``monte_carlo_pingpong.cl`` kernel (ping-pong tree
reduction over two local scratch buffers).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)

class PyOpenCLMonteCarloPingPongAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter defaulting to the ``monte_carlo_pingpong.cl`` sampling kernel."""

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
