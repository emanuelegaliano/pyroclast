"""Monte Carlo Service — orchestrates the simulation stage.

This module is the single entry-point for running the Monte Carlo
probability estimation.  It sits in the Service Layer of the Ports &
Adapters architecture: it depends only on abstract Ports
(:class:`~pyroclast.ABCs.monte_carlo.IMonteCarloAdapter`) and domain
Value Objects (:class:`~pyroclast.domain.models.CompactedHabitat`,
:class:`~pyroclast.domain.models.MonteCarloConfig`).

Degenerate case
---------------
A :class:`~pyroclast.domain.models.CompactedHabitat` with ``n_cells=0``
means the habitat category is absent from the study area.  The probability
of exceeding any threshold is ``0.0`` by definition; the adapter is never
called for such habitats to avoid division-by-zero inside the kernel.

See also
--------
pyroclast.ABCs.monte_carlo.IMonteCarloAdapter : compute Port.
pyroclast.domain.models.MonteCarloConfig : simulation parameters.
pyroclast.domain.models.CompactedHabitat : input Value Object.
"""

from __future__ import annotations

import logging

from pyroclast.ABCs.monte_carlo import IMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig

logger = logging.getLogger(__name__)


def run_monte_carlo(
    adapter: IMonteCarloAdapter,
    habitat: CompactedHabitat,
    config: MonteCarloConfig,
) -> float:
    """Estimate the lava-destruction probability for a single habitat.

    Delegates to ``adapter.run`` after guarding against the degenerate
    empty-habitat case (``n_cells=0``).

    Parameters
    ----------
    adapter : IMonteCarloAdapter
        Compute backend (GPU, CPU, or stub).
    habitat : CompactedHabitat
        Pre-processed habitat produced by the preprocessing pipeline.
    config : MonteCarloConfig
        Simulation parameters: number of runs, threshold, and seed.

    Returns
    -------
    float
        Estimated probability in ``[0.0, 1.0]`` that the invaded fraction
        of habitat cells **strictly exceeds** ``config.threshold``.
        Returns ``0.0`` immediately if ``habitat.n_cells == 0``.

    Examples
    --------
    >>> prob = run_monte_carlo(adapter, habitat, config)
    >>> assert 0.0 <= prob <= 1.0
    """
    if habitat.n_cells == 0:
        logger.debug(
            "run_monte_carlo: habitat '%s' has n_cells=0 — returning 0.0.",
            habitat.habitat_code,
        )
        return 0.0

    prob = adapter.run(habitat, config)
    logger.info(
        "run_monte_carlo: habitat '%s' — prob=%.4f (R=%d, theta=%.3f).",
        habitat.habitat_code,
        prob,
        config.n_runs,
        config.threshold,
    )
    return prob


def run_monte_carlo_batch(
    adapter: IMonteCarloAdapter,
    habitats: list[CompactedHabitat],
    config: MonteCarloConfig,
) -> dict[str, float]:
    """Estimate destruction probability for a collection of habitats.

    Calls :func:`run_monte_carlo` for each habitat and collects the results
    in a dictionary keyed by ``habitat_code``.  Habitats with ``n_cells=0``
    yield ``0.0`` without invoking the adapter.

    Parameters
    ----------
    adapter : IMonteCarloAdapter
        Compute backend shared across all habitats.
    habitats : list[CompactedHabitat]
        Habitats to simulate.  An empty list returns an empty dict.
    config : MonteCarloConfig
        Simulation parameters applied uniformly to every habitat.

    Returns
    -------
    dict[str, float]
        Mapping from ``habitat_code`` to estimated probability.  Keys are
        in the same order as the input ``habitats`` list.

    Examples
    --------
    >>> results = run_monte_carlo_batch(adapter, habitats, config)
    >>> prob_9340 = results["9340"]
    """
    return {
        h.habitat_code: run_monte_carlo(adapter, h, config)
        for h in habitats
    }
