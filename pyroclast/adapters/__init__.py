"""Adapters (infrastructure) layer for pyroclast.

Contains concrete implementations of the compute Ports defined in
``pyroclast.ABCs.compute``.  Each adapter isolates a specific technology
(PyOpenCL, CUDA, NumPy-CPU, …) from the rest of the application.
"""

from .opencl_adapter import PyOpenCLAdapter
from .opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from .opencl_mc_2d_stride_adapter import PyOpenCLMonteCarlo2DAdapter
from .opencl_mc_2d_pingpong_adapter import PyOpenCLMonteCarlo2DPingPongAdapter
from .opencl_mc_2d_two_barriers_adapter import PyOpenCLMonteCarlo2DTwoBarriersAdapter
from .opencl_mc_pingpong_adapter import PyOpenCLMonteCarloPingPongAdapter
from .opencl_mc_vectorized_adapter import PyOpenCLMonteCarloVectorizedAdapter
from .opencl_mc_vectorized_pingpong_adapter import (
    PyOpenCLMonteCarloVectorizedPingPongAdapter,
)

__all__ = [
    "PyOpenCLAdapter",
    "PyOpenCLMonteCarloAdapter",
    "PyOpenCLMonteCarlo2DAdapter",
    "PyOpenCLMonteCarlo2DPingPongAdapter",
    "PyOpenCLMonteCarlo2DTwoBarriersAdapter",
    "PyOpenCLMonteCarloPingPongAdapter",
    "PyOpenCLMonteCarloVectorizedAdapter",
    "PyOpenCLMonteCarloVectorizedPingPongAdapter",
]
