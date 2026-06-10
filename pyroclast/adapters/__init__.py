"""Adapters (infrastructure) layer for pyroclast.

Contains concrete implementations of the compute Ports defined in
``pyroclast.ABCs.compute``.  Each adapter isolates a specific technology
(PyOpenCL, CUDA, NumPy-CPU, …) from the rest of the application.
"""

from .opencl_adapter import (
    PyOpenCLAdapter,
    PyOpenCLHostScalarCompactionAdapter,
    PyOpenCLHostNonzeroCompactionAdapter,
    PyOpenCLHostCompressCompactionAdapter,
    PyOpenCLGPUScalarCompactionAdapter,
    PyOpenCLGPUVectorizedCompactionAdapter,
)
from .opencl_mc_2d_adapter import PyOpenCLMonteCarlo2DAdapter
from .opencl_mc_2d_transposed_adapter import PyOpenCLMonteCarlo2DTransposedAdapter
from .opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from .opencl_mc_commutative_adapter import PyOpenCLMonteCarloCommutativeAdapter
from .opencl_mc_global_seed_adapter import PyOpenCLMonteCarloGlobalSeedAdapter
from .opencl_mc_map_centric_adapter import PyOpenCLMapCentricAdapter
from .opencl_mc_pingpong_adapter import PyOpenCLMonteCarloPingPongAdapter
from .opencl_mc_vectorized_adapter import PyOpenCLMonteCarloVectorizedAdapter
from .opencl_mc_vectorized_pingpong_adapter import (
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
)
from .opencl_mc_contiguous_adapter import PyOpenCLMonteCarloContiguousAdapter

__all__ = [
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
    "PyOpenCLMonteCarloContiguousAdapter",
]

