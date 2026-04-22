"""Monte Carlo Port — abstract interface for the simulation backend.

This module defines the *Port* through which the Service Layer requests
the Monte Carlo probability estimate without depending on any specific
compute technology (PyOpenCL, CUDA, CPU NumPy, …).

Architectural role
------------------
``IMonteCarloAdapter`` is a **driven port** (secondary / right-side port):
the application core calls it to delegate the stochastic simulation to an
external compute backend.  Any class that implements this interface is an
*Adapter* and lives in ``pyroclast/adapters/``.

The port intentionally exposes a single, coarse-grained operation — one
method per habitat — so that the Service Layer remains unaware of how
parallelism is organised internally (NDRange shape, batch size, RNG
algorithm, etc.).

See also
--------
pyroclast.adapters.opencl_mc_adapter.PyOpenCLMonteCarloAdapter : GPU impl.
pyroclast.services.monte_carlo.run_monte_carlo : primary consumer.
pyroclast.domain.models.MonteCarloConfig : simulation parameters.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from pyroclast.domain.models import BenchResult, CompactedHabitat, MonteCarloConfig


class IMonteCarloAdapter(ABC):
    """Abstract Port for the Monte Carlo simulation stage.

    An ``IMonteCarloAdapter`` receives a pre-processed
    :class:`~pyroclast.domain.models.CompactedHabitat` and the simulation
    configuration, runs :math:`R` independent stochastic trials, and returns
    the estimated probability that the fraction of invaded habitat cells
    **strictly exceeds** the critical threshold :math:`\\theta`.

    The caller guarantees that ``habitat.n_cells > 0``; the degenerate
    zero-cell case is handled by
    :func:`~pyroclast.services.monte_carlo.run_monte_carlo` before this
    method is invoked.

    Implementations are responsible for:

    * Allocating and releasing all device-side buffers.
    * Generating uniform random samples with the strategy embedded in the
      concrete adapter (device-side RNG seeded from ``config.seed``, or any
      other reproducible mechanism).
    * Returning a value in :math:`[0.0, 1.0]` consistent with the Monte Carlo
      estimate ``count(over_threshold) / R``.

    Notes
    -----
    Reproducibility: given identical ``habitat`` and ``config`` arguments, the
    same probability estimate **must** be returned across calls.  This is
    required for deterministic testing and scientific reproducibility.

    Examples
    --------
    Typical usage inside the service layer::

        adapter: IMonteCarloAdapter = PyOpenCLMonteCarloAdapter()
        prob = adapter.run(habitat, config)
    """

    @abstractmethod
    def run(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
    ) -> float:
        """Estimate the destruction probability for a single habitat.

        Runs ``config.n_runs`` independent Monte Carlo trials.  In each trial
        every active cell :math:`k` is independently invaded with probability
        ``habitat.p_vec[k]``; the trial is a *destruction event* when the
        invaded fraction strictly exceeds ``config.threshold``.

        Parameters
        ----------
        habitat : CompactedHabitat
            Pre-processed habitat.  ``habitat.n_cells`` must be > 0; the
            caller is responsible for this precondition.
        config : MonteCarloConfig
            Simulation parameters: number of runs, threshold, and seed.

        Returns
        -------
        float
            Estimated probability in :math:`[0.0, 1.0]`, equal to
            ``count(over_threshold) / config.n_runs``.

        Raises
        ------
        RuntimeError
            If device initialisation or kernel execution fails (raised by the
            concrete implementation).
        """

    @abstractmethod
    def run_batched(
        self,
        habitat: CompactedHabitat,
        config: MonteCarloConfig,
        n_batches: int,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> float:
        """Estimate destruction probability running config.n_runs in n_batches.

        Uploads habitat.p_vec once and keeps it in device memory for the
        entire run, downloading only a single int32 per batch.

        Parameters
        ----------
        habitat : CompactedHabitat
            Pre-processed habitat.  ``habitat.n_cells`` must be > 0.
        config : MonteCarloConfig
            Simulation parameters.  ``config.n_runs`` is the total number of
            trials across all batches.
        n_batches : int
            Number of kernel launches.  ``config.n_runs`` must be divisible
            by ``n_batches``.
        callback : callable, optional
            Called after each batch as ``callback(batch_index, n_batches,
            partial_prob)`` where ``partial_prob`` is the running estimate.

        Returns
        -------
        float
            Estimated probability in :math:`[0.0, 1.0]`.
        """

    def benchmark(self) -> BenchResult:
        """Return timing and bandwidth statistics from real kernel executions.

        Statistics are collected during actual ``run`` or ``run_batched``
        calls when the adapter is constructed with ``profiling=True``.

        Returns
        -------
        BenchResult
            Timing and bandwidth statistics derived from real kernel launches.

        Raises
        ------
        NotImplementedError
            If the concrete adapter does not support profiling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support benchmark()."
        )
