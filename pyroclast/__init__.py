from .io import FileMapRepository, FileMapStrategy, GeoTiffMap, HabitatCriteria, InvasionCriteria
from .ABCs import MapRepository, MapCriteria, RasterMap, MapRepositoryStrategy, IComputeAdapter
from .domain import CompactedHabitat, GridTopology, SpatialHabitat
from .adapters import (
    PyOpenCLAdapter,
    PyOpenCLHostScalarCompactionAdapter,
    PyOpenCLHostNonzeroCompactionAdapter,
    PyOpenCLHostCompressCompactionAdapter,
    PyOpenCLGPUScalarCompactionAdapter,
    PyOpenCLGPUVectorizedCompactionAdapter,
    PyOpenCLMapCentricAdapter,
    PyOpenCLMonteCarlo2DAdapter,
    PyOpenCLMonteCarlo2DTransposedAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloCommutativeAdapter,
    PyOpenCLMonteCarloGlobalSeedAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
    PyOpenCLMonteCarloVectorizedAdapter,
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
)
from .services import run_preprocessing_batch, generate_synthetic_habitat_dem

__all__ = [
    # Repository layer
    "FileMapRepository",
    "HabitatCriteria",
    "InvasionCriteria",
    "GeoTiffMap",
    "MapRepository",
    "MapCriteria",
    "RasterMap",
    "MapRepositoryStrategy",
    # Compute layer
    "IComputeAdapter",
    "PyOpenCLAdapter",
    "PyOpenCLHostScalarCompactionAdapter",
    "PyOpenCLHostNonzeroCompactionAdapter",
    "PyOpenCLHostCompressCompactionAdapter",
    "PyOpenCLGPUScalarCompactionAdapter",
    "PyOpenCLGPUVectorizedCompactionAdapter",
    "PyOpenCLMonteCarloAdapter",
    "PyOpenCLMonteCarloCommutativeAdapter",
    "PyOpenCLMonteCarloGlobalSeedAdapter",
    "PyOpenCLMonteCarloPingPongAdapter",
    "PyOpenCLMonteCarloVectorizedAdapter",
    "PyOpenCLMonteCarloVectorizedPingPongAdapter",
    "PyOpenCLMapCentricAdapter",
    "PyOpenCLMonteCarlo2DAdapter",
    "PyOpenCLMonteCarlo2DTransposedAdapter",
    # Domain
    "CompactedHabitat",
    "GridTopology",
    "SpatialHabitat",
    # Services
    "run_preprocessing_batch",
    "generate_synthetic_habitat_dem",
]

