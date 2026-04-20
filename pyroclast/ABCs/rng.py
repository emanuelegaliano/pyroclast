"""RNG Port — abstract strategy for uniform random sample generation.

This module defines the *Port* through which the Monte Carlo service requests
uniform random samples without knowing whether they are produced on the CPU,
by a device-side generator, or by any other backend.

Architectural role
------------------
``IRngStrategy`` is a **driven port** in the Ports & Adapters architecture.
The Service Layer calls it to obtain samples; the concrete implementation
(the *Adapter*) decides how and where the numbers are generated.

Swapping from a host-side NumPy generator to a GPU-native PRNG (e.g. MWC64X,
Philox) requires only a different ``IRngStrategy`` implementation — the Service
Layer and domain code are unchanged.

Any class that implements ``IRngStrategy`` lives in ``pyroclast/adapters/``.

See also
--------
pyroclast.domain.models.MonteCarloConfig : carries the seed and run count.
"""

from abc import ABC, abstractmethod

import numpy as np


class IRngStrategy(ABC):
    """Abstract Port for generating uniform random samples for Monte Carlo simulation.

    An ``IRngStrategy`` produces a 2-D array of independent samples drawn from
    :math:`U(0, 1)`, shaped ``(n_runs, n_cells)``.  Each row corresponds to one
    Monte Carlo run; each column corresponds to one compacted habitat cell.

    The Service Layer uses these samples directly in the invasion test::

        invaded[r, k] = sample[r, k] <= p_vec[k]

    Implementations may generate samples on the CPU (e.g. via ``numpy.random``),
    on the GPU kernel side (returning a pre-copied host array), or via any
    other backend.

    Notes
    -----
    Implementations must guarantee:

    * Output dtype is ``numpy.float32``.
    * All values are in the half-open interval ``[0.0, 1.0)``.
    * Given the same ``seed``, ``n_runs``, and ``n_cells``, the output is
      identical across calls (*reproducibility*).

    Examples
    --------
    Typical usage inside the service layer::

        rng: IRngStrategy = NumpyRngAdapter()
        samples = rng.generate(n_runs=config.n_runs, n_cells=habitat.n_cells, seed=config.seed)
    """

    @abstractmethod
    def generate(self, n_runs: int, n_cells: int, seed: int) -> np.ndarray:
        """Generate a ``(n_runs, n_cells)`` array of :math:`U(0,1)` samples.

        Parameters
        ----------
        n_runs : int
            Number of Monte Carlo runs (rows).  Must be > 0.
        n_cells : int
            Number of compacted habitat cells (columns).  Must be > 0.
        seed : int
            Deterministic seed.  The same seed always produces the same output.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_runs, n_cells)``, dtype ``float32``, values in
            ``[0.0, 1.0)``.

        Raises
        ------
        ValueError
            If ``n_runs <= 0`` or ``n_cells <= 0``.
        """
