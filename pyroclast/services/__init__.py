"""Service layer for pyroclast.

Orchestrates domain objects, repositories, and compute adapters.
Service functions are the single entry-points used by CLI, notebooks, or tests.
"""

from .preprocessing import run_preprocessing_batch
from .monte_carlo import run_monte_carlo, run_monte_carlo_batch
from .synthetic import generate_synthetic_habitat_dem, generate_synthetic_dem

__all__ = [
    "run_preprocessing_batch",
    "run_monte_carlo",
    "run_monte_carlo_batch",
    "generate_synthetic_habitat_dem",
    "generate_synthetic_dem",
]
