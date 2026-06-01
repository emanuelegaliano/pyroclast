"""PyOpenCL Monte Carlo Commutative Adapter.

Adapter for the ``monte_carlo_commutative.cl`` kernel: the same scalar
sampling as :class:`PyOpenCLMonteCarloAdapter` (``_run_trial`` via
``misc.h``, MWC64X RNG), but with the *commutative* work-group reduction:

* A single local buffer ``lmem[WG_SIZE]`` replaces the two-buffer
  ping-pong approach of :class:`PyOpenCLMonteCarloPingPongAdapter`.
* One ``barrier(CLK_LOCAL_MEM_FENCE)`` is issued at the **top** of each
  halving step (instead of at the bottom).
* Active lanes accumulate the high-half partner into a private register
  ``val`` and write it back to ``lmem[lid]`` — no pointer-swap needed.
* The thread leader writes ``val`` (not ``lmem[0]``) to ``partial[group_id]``.

The sampling, topology suggestion, recursive reducer and profiling API are
all inherited unchanged from :class:`PyOpenCLMonteCarloAdapter`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyroclast.adapters.opencl_mc_adapter import PyOpenCLMonteCarloAdapter

logger = logging.getLogger(__name__)


class PyOpenCLMonteCarloCommutativeAdapter(PyOpenCLMonteCarloAdapter):
    """Adapter defaulting to the ``monte_carlo_commutative.cl`` sampling kernel.

    Drop-in replacement for :class:`PyOpenCLMonteCarloAdapter` and
    :class:`PyOpenCLMonteCarloPingPongAdapter`: same public API, same
    ``run()`` / ``run_batched()`` / ``benchmark()`` interface.
    """

    def __init__(
        self,
        kernel_path: Path | None = None,
        profiling: bool = False,
    ) -> None:
        if kernel_path is None:
            kernel_path = (
                Path(__file__).parent.parent
                / "kernels"
                / "monte_carlo_commutative.cl"
            )
        super().__init__(kernel_path=kernel_path, profiling=profiling)
        logger.info(
            "PyOpenCLMonteCarloCommutativeAdapter initialized with kernel: %s",
            Path(kernel_path).name,
        )
