from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class MapCriteria(ABC):
    """Query predicate passed to the repository."""


class RasterMap(ABC):
    """A single raster layer — the domain object."""

    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @property
    @abstractmethod
    def data(self) -> np.ndarray: ...

    @abstractmethod
    def satisfies(self, criteria: MapCriteria) -> bool: ...


class MapRepositoryStrategy(ABC):
    """Holds the actual collection of maps and filters them."""

    @abstractmethod
    def matching(self, criteria: MapCriteria) -> Sequence[RasterMap]: ...


class MapRepository(ABC):
    """Entry point for clients — delegates to a MapRepositoryStrategy."""

    @abstractmethod
    def matching(self, criteria: MapCriteria) -> Sequence[RasterMap]: ...

    def get(self, criteria: MapCriteria) -> RasterMap:
        """Return the single map matching criteria. Raises if not exactly one."""
        results = self.matching(criteria)
        if len(results) != 1:
            raise ValueError(
                f"Expected exactly one result for {criteria!r}, got {len(results)}"
            )
        return results[0]
