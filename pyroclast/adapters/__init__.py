"""Adapters (infrastructure) layer for pyroclast.

Contains concrete implementations of the compute Ports defined in
``pyroclast.ABCs.compute``.  Each adapter isolates a specific technology
(PyOpenCL, CUDA, NumPy-CPU, …) from the rest of the application.
"""

from .opencl_adapter import PyOpenCLAdapter
from .opencl_mc_adapter import PyOpenCLMonteCarloAdapter
from .opencl_mc_2d_stride_adapter import PyOpenCLMonteCarlo2DAdapter
from .opencl_mc_2d_pingpong_adapter import PyOpenCLMonteCarlo2DPingPongAdapter
from .opencl_mc_pingpong_adapter import PyOpenCLMonteCarloPingPongAdapter
from .opencl_mc_cascading_adapter import PyOpenCLMonteCarloCascadingAdapter

__all__ = [
    "PyOpenCLAdapter",
    "PyOpenCLMonteCarloAdapter",
    "PyOpenCLMonteCarlo2DAdapter",
    "PyOpenCLMonteCarlo2DPingPongAdapter",
    "PyOpenCLMonteCarloPingPongAdapter",
    "PyOpenCLMonteCarloCascadingAdapter",
]
