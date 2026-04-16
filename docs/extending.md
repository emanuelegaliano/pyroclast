# Data Loading — Usage, Modification and Extension

The data loading layer is built around the **Repository pattern**. The full class diagram is in [patterns.md](patterns.md).

The relevant module is `pyroclast/io/data_repository.py`, with abstract base classes in `pyroclast/ABCs/repository.py`.

---

## Basic usage

```python
from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria

repo = FileMapRepository("data")

# Single result — use get()
invasion = repo.get(InvasionCriteria())
p = invasion.data                          # np.ndarray float32, shape (n, n)

# Multiple results — use matching()
habitats = repo.matching(HabitatCriteria())        # all habitat maps
oak = repo.get(HabitatCriteria(code="9340"))       # one specific habitat
```

`data` on any `RasterMap` is a numpy array ready for GPU upload — no NaN, no post-processing needed.

---

## Adding a new habitat

Drop a file named `cb_codice_<code>.tif` into `data/habitats/`. The repository discovers it automatically on the next instantiation — no code changes required.

The only constraint is that the new file must have the same shape and CRS as the invasion map. The strategy validates this on load and raises `ValueError` if they differ.

---

## Adding a new criteria type

### How it works

A criteria object carries the parameters of a query. The repository itself has no filtering logic — it delegates to each map's `satisfies()` method and collects the matches:

```python
# FileMapStrategy.matching() — no filtering logic here
return [m for m in self._all_maps if m.satisfies(criteria)]
```

`GeoTiffMap.satisfies()` then checks which criteria it receives and decides accordingly:

```python
def satisfies(self, criteria: MapCriteria) -> bool:
    if isinstance(criteria, HabitatCriteria):
        ...
    if isinstance(criteria, InvasionCriteria):
        ...
```

Adding a new query type never touches the repository — only a new class and a new `isinstance` branch in `satisfies()`.

### Steps

1. Define a new class inheriting from `MapCriteria`:

```python
from pyroclast.ABCs.repository import MapCriteria
from dataclasses import dataclass

@dataclass(frozen=True)
class HighRiskCriteria(MapCriteria):
    """Matches maps that have at least `min_cells` cells with P > threshold."""
    threshold: float = 0.3
    min_cells: int = 1000
```

2. Implement `satisfies()` in `GeoTiffMap` for the new criteria type:

```python
# in GeoTiffMap.satisfies()
if isinstance(criteria, HighRiskCriteria):
    import numpy as np
    return int(np.sum(self._data > criteria.threshold)) >= criteria.min_cells
```

3. Use it like any other criteria:

```python
high_risk = repo.matching(HighRiskCriteria(threshold=0.3, min_cells=5000))
```

---

## Replacing the data source

To load maps from a different source (e.g. a remote WCS, a database, or synthetic data for testing) implement a new `MapRepositoryStrategy`:

```python
from pyroclast.ABCs.repository import MapRepositoryStrategy, MapCriteria, RasterMap
from collections.abc import Sequence

class SyntheticMapStrategy(MapRepositoryStrategy):
    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = shape

    def matching(self, criteria: MapCriteria) -> Sequence[RasterMap]:
        # build and return synthetic GeoTiffMap objects
        ...
```

Then wire it into a custom repository:

```python
from pyroclast.ABCs.repository import MapRepository

class SyntheticMapRepository(MapRepository):
    def __init__(self, shape):
        self._strategy = SyntheticMapStrategy(shape)

    def matching(self, criteria):
        return self._strategy.matching(criteria)
```

The rest of the codebase — pipeline, GPU kernels, tests — only depend on `MapRepository` and `RasterMap`, so the swap is transparent.

---

## Data conventions

| Map type | dtype | Expected values | Nodata |
|---|---|---|---|
| Invasion (`mappa_probabilità*.tif`) | `float32` | `[0.0, 1.0]` | `NaN` → converted to `0.0` on load |
| Habitat (`cb_codice_*.tif`) | `uint8` | `{0, 1}` | `0` (absence) |
