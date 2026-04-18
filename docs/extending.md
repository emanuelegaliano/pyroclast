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

## Replacing the Compute Adapter

The GPU preprocessing backend is isolated behind the `IComputeAdapter` Port
defined in `pyroclast/ABCs/compute.py`. Swapping the implementation — for
example to use CUDA instead of OpenCL, or to run on CPU for testing — requires
implementing a single class and injecting it into the service.

### How it works

`run_preprocessing_batch` never imports `PyOpenCLAdapter` directly:

```python
def run_preprocessing_batch(
    repo: MapRepository,
    compute: IComputeAdapter,   # ← the Port, not a concrete class
    criteria: MapCriteria,
    cache_dir: Path,
) -> list[CompactedHabitat]: ...
```

Any class that implements `IComputeAdapter.batch_preprocess()` can be passed
here — the service layer does not care about the technology underneath.

### Implementing a CudaAdapter (example)

```python
# pyroclast/adapters/cuda_adapter.py
import cupy as cp
from collections.abc import Sequence
from pyroclast.ABCs.compute import IComputeAdapter
from pyroclast.ABCs.repository import RasterMap
from pyroclast.domain.models import CompactedHabitat


class CudaAdapter(IComputeAdapter):
    """Compute adapter backed by CuPy/CUDA."""

    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        p_gpu = cp.asarray(invasion_map.data.ravel().astype("float32"))
        results = []
        for habitat in habitats:
            h_gpu = cp.asarray(habitat.data.ravel().astype("uint8"))
            out_gpu = p_gpu * h_gpu.astype("float32")
            out_cpu = cp.asnumpy(out_gpu)
            mask = out_cpu > 0.0
            p_vec = out_cpu[mask].copy()
            results.append(
                CompactedHabitat(
                    habitat_code=habitat.code,
                    n_cells=len(p_vec),
                    p_vec=p_vec,
                )
            )
        return results
```

Wire it in at the application entry-point:

```python
from pyroclast.adapters.cuda_adapter import CudaAdapter
from pyroclast.services import run_preprocessing_batch

results = run_preprocessing_batch(
    repo=repo,
    compute=CudaAdapter(),   # ← swap here, nothing else changes
    criteria=HabitatCriteria(),
    cache_dir=Path("cache/"),
)
```

### Implementing a NumPyCpuAdapter (testing / CI)

For unit tests or machines without a GPU, a pure-NumPy CPU implementation
fulfils the same interface:

```python
# tests/stubs.py
import numpy as np
from collections.abc import Sequence
from pyroclast.ABCs.compute import IComputeAdapter
from pyroclast.ABCs.repository import RasterMap
from pyroclast.domain.models import CompactedHabitat


class NumPyCpuAdapter(IComputeAdapter):
    """CPU fallback — identical semantics to PyOpenCLAdapter, no GPU needed."""

    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        p = invasion_map.data.ravel().astype(np.float32)
        results = []
        for habitat in habitats:
            out = p * habitat.data.ravel().astype(np.float32)
            mask = out > 0.0
            p_vec = out[mask].copy()
            results.append(
                CompactedHabitat(
                    habitat_code=habitat.code,
                    n_cells=len(p_vec),
                    p_vec=p_vec,
                )
            )
        return results
```

The test suite can then inject `NumPyCpuAdapter()` without needing any OpenCL
runtime:

```python
def test_preprocessing_batch(tmp_path):
    repo = SyntheticMapRepository(shape=(64, 64))
    compute = NumPyCpuAdapter()
    results = run_preprocessing_batch(
        repo=repo,
        compute=compute,
        criteria=HabitatCriteria(),
        cache_dir=tmp_path,
    )
    assert len(results) > 0
```

---

## Caching mechanism in the Service Layer

`run_preprocessing_batch` implements an **incremental disk cache** to avoid
redundant GPU work across runs.

### Cache layout

```
cache/
├── habitat_92XX.npy
├── habitat_9340.npy
└── habitat_9530_.npy
```

Each file stores the compacted `p_vec` array (1-D `float32`) for one habitat.
The filename encodes the habitat code; `n_cells` is re-derived from
`len(p_vec)` on load.

### Cache logic

On each call to `run_preprocessing_batch`:

1. All habitats matching `criteria` are retrieved from the repository.
2. For each habitat the service checks for ``cache_dir/habitat_<code>.npy``.
3. **Cache hits** are loaded with `np.load()` — no GPU call is made.
4. **Cache misses** are collected into a batch and forwarded to
   `compute.batch_preprocess()` in a single call.
5. New results are saved to ``cache_dir`` with `np.save()`.
6. The full list (cached + new) is returned sorted by `habitat_code`.

### Invalidating the cache

To force recomputation, simply delete the relevant `.npy` files:

```bash
# Invalidate all habitats
rm cache/habitat_*.npy

# Invalidate a specific habitat
rm cache/habitat_9340.npy
```

The next call to `run_preprocessing_batch` will recompute only the deleted
entries and repopulate the cache.

### Thread safety

The current implementation is **not** thread-safe.  If multiple processes or
threads call `run_preprocessing_batch` concurrently with the same `cache_dir`,
a race condition on cache writes is possible.  For parallel workloads, use a
separate `cache_dir` per process or add file locking (e.g. `filelock`).

---

## Data conventions

| Map type | dtype | Expected values | Nodata |
|---|---|---|---|
| Invasion (`mappa_probabilità*.tif`) | `float32` | `[0.0, 1.0]` | `NaN` → converted to `0.0` on load |
| Habitat (`cb_codice_*.tif`) | `uint8` | `{0, 1}` | `0` (absence) |
