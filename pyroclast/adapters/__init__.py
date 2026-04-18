"""Adapters (infrastructure) layer for pyroclast.

Contains concrete implementations of the compute Ports defined in
``pyroclast.ABCs.compute``.  Each adapter isolates a specific technology
(PyOpenCL, CUDA, NumPy-CPU, …) from the rest of the application.
"""

from .opencl_adapter import PyOpenCLAdapter

__all__ = ["PyOpenCLAdapter"]
