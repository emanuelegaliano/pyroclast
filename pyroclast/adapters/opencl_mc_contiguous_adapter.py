"""PyOpenCL Monte Carlo Contiguous Adapter.

Adapter for the ``monte_carlo_contiguous.cl`` kernel and its multi-habitat variant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarloContiguousAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter for the contiguous sliding-window Monte Carlo kernels."""

    _SAMPLING_KERNEL_NAME = "monte_carlo_contiguous"
    _MULTI_SAMPLING_KERNEL_NAME = "monte_carlo_contiguous_multi"

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
        extra_build_options: str = "",
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo"
                / "monte_carlo_contiguous.cl"
            )
        super().__init__(
            kernel_path=kernel_path,
            profiling=profiling,
            extra_build_options=extra_build_options,
        )
        logger.info(
            "PyOpenCLMonteCarloContiguousAdapter initialized with kernel: %s",
            Path(kernel_path).name,
        )
