"""Domain layer for pyroclast.

Contains pure Value Objects that represent the core concepts of the simulation.
No infrastructure dependencies are allowed in this package.
"""

from .models import CompactedHabitat, MonteCarloConfig

__all__ = ["CompactedHabitat", "MonteCarloConfig"]
