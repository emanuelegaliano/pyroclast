"""PyOpenCL Monte Carlo Cascading Adapter — optimized GPU reduction.

This adapter uses the ``monte_carlo_cascading.cl`` kernel, which implements
a grid-stride loop and a sequential addressing reduction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)

class PyOpenCLMonteCarloCascadingAdapter(PyOpenCLMonteCarloAdapter):
    """GPU Monte Carlo adapter using a cascading reduction kernel."""

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent / "kernels" / "monte_carlo_cascading.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)
        logger.info(
            "PyOpenCLMonteCarloCascadingAdapter initialized with kernel: %s",
            kernel_path.name,
        )
