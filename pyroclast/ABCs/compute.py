"""Compute Port — abstract interface for GPU/CPU preprocessing backends.

This module defines the *Port* through which the Service Layer communicates
with any compute back-end.  Following the Ports & Adapters (Hexagonal)
architecture, the Service Layer depends **only** on this interface; it never
imports PyOpenCL, CUDA, or any hardware-specific library directly.

Architectural role
------------------
*Ports* sit on the boundary between the application core (domain + services)
and the infrastructure.  ``IComputeAdapter`` is a **driven port** (also called
a *secondary* or *right-side* port): the application calls it to delegate
work to an external system (the GPU).

Any class that implements ``IComputeAdapter`` is an *Adapter* and lives in
``pyroclast/adapters/``.  Swapping from PyOpenCL to CUDA, Metal, or a pure
NumPy CPU fallback requires only a different ``IComputeAdapter`` implementation
— the Service Layer code is unchanged.

See also
--------
pyroclast.adapters.opencl_adapter.PyOpenCLAdapter : reference GPU implementation.
pyroclast.services.preprocessing.run_preprocessing_batch : primary consumer.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pyroclast.ABCs.repository import RasterMap
from pyroclast.domain.models import BenchResult, CompactedHabitat


class IComputeAdapter(ABC):
    """Abstract Port for batch preprocessing of habitat rasters on a compute device.

    An ``IComputeAdapter`` receives the invasion-probability raster and a
    collection of habitat rasters, executes the element-wise multiplication
    (Map kernel) on a compute device, performs stream compaction to discard
    zero cells, and returns a list of :class:`~pyroclast.domain.models.CompactedHabitat`
    Value Objects ready for the Monte Carlo simulation.

    Implementations are responsible for:

    * Device initialisation (context, command queue, kernel compilation).
    * Memory management: allocation, host→device transfers, device→host
      readback, and deallocation.
    * Stream compaction: filtering cells with probability > 0.

    The invasion map buffer **should** be transferred to the device only once
    per ``batch_preprocess`` call, even when multiple habitats are processed.

    Notes
    -----
    Implementors must be careful to release all device-side buffers before
    returning, to avoid VRAM leaks when the service layer calls this method
    repeatedly (e.g. in a batch loop with partial caching).

    Examples
    --------
    Typical usage inside the service layer::

        adapter: IComputeAdapter = PyOpenCLAdapter()
        invasion = repo.get(InvasionCriteria())
        habitats = repo.matching(HabitatCriteria())
        results = adapter.batch_preprocess(invasion, habitats)
    """

    @abstractmethod
    def batch_preprocess(
        self,
        invasion_map: RasterMap,
        habitats: Sequence[RasterMap],
    ) -> list[CompactedHabitat]:
        """Execute the Map kernel + stream compaction for a batch of habitats.

        Transfers the ``invasion_map`` to device memory once, then iterates
        over ``habitats``: for each one it copies the habitat raster to the
        device, runs the ``map_multiply`` kernel, reads back the result, and
        applies stream compaction (retaining only cells with value > 0).

        The output list is in the **same order** as the input ``habitats``
        sequence, enabling callers to zip inputs and outputs without
        additional bookkeeping.

        Parameters
        ----------
        invasion_map : RasterMap
            The invasion-probability raster.  ``invasion_map.data`` must be a
            2-D ``numpy.ndarray`` of dtype ``float32`` with values in
            ``[0.0, 1.0]``.  NaN values must have been replaced with ``0.0``
            before this call (guaranteed by ``FileMapStrategy``).
        habitats : Sequence[RasterMap]
            Ordered sequence of habitat-presence rasters.  Each
            ``habitat.data`` must be a 2-D ``numpy.ndarray`` of dtype
            ``uint8`` with the same shape as ``invasion_map.data``.
            An empty sequence is valid and returns an empty list.

        Returns
        -------
        list[CompactedHabitat]
            One :class:`~pyroclast.domain.models.CompactedHabitat` per
            element of ``habitats``, in the same order.  A habitat with no
            active cells produces a ``CompactedHabitat`` with ``n_cells=0``
            and an empty ``p_vec``.

        Raises
        ------
        ValueError
            If the shapes of ``invasion_map.data`` and any ``habitat.data``
            do not match, or if dtype requirements are not satisfied.
        RuntimeError
            If device initialisation or kernel compilation fails (raised by
            the concrete implementation).
        """

    def benchmark(self) -> BenchResult:
        """Return timing and bandwidth statistics from real kernel executions.

        Statistics are collected during actual ``batch_preprocess`` calls when
        the adapter is constructed with ``profiling=True``.  Calling this
        method without having run at least one ``batch_preprocess`` first, or
        without enabling profiling, raises ``NotImplementedError``.

        Returns
        -------
        BenchResult
            Timing and bandwidth statistics derived from real kernel launches.

        Raises
        ------
        NotImplementedError
            If the concrete adapter does not support profiling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support benchmark()."
        )
