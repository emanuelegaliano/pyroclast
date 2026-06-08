"""PyOpenCL Monte Carlo Global Seeding Adapter.

Adapter for the ``monte_carlo_global_seed.cl`` kernel: work-item level RNG seeding
combined with commutative work-group reduction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarloGlobalSeedAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter defaulting to the ``monte_carlo_global_seed.cl`` sampling kernel.

    RNG streams are seeded once at work-item level, yielding substantial speedups
    by avoiding reseeding overhead across runs and habitats.
    """

    _SAMPLING_KERNEL_NAME = "monte_carlo_global_seed"
    _MULTI_SAMPLING_KERNEL_NAME = "monte_carlo_global_seed_multi"

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
                / "monte_carlo_global_seed.cl"
            )
        super().__init__(
            kernel_path=kernel_path,
            profiling=profiling,
            extra_build_options=extra_build_options,
        )
        logger.info(
            "PyOpenCLMonteCarloGlobalSeedAdapter initialized with kernel: %s",
            Path(kernel_path).name,
        )
