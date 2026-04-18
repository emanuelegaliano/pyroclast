"""Service layer for pyroclast.

Orchestrates domain objects, repositories, and compute adapters.
Service functions are the single entry-points used by CLI, notebooks, or tests.
"""

from .preprocessing import run_preprocessing_batch

__all__ = ["run_preprocessing_batch"]
