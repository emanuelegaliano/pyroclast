import numpy as np
import pytest

from pyroclast import (
    CompactedHabitat,
    PyOpenCLAdapter,
    PyOpenCLHostScalarCompactionAdapter,
    PyOpenCLHostNonzeroCompactionAdapter,
    PyOpenCLHostCompressCompactionAdapter,
    PyOpenCLGPUScalarCompactionAdapter,
    PyOpenCLGPUVectorizedCompactionAdapter,
)
from pyroclast.io.data_repository import GeoTiffMap


@pytest.fixture(scope="module")
def adapters():
    classes = [
        ("Baseline", PyOpenCLAdapter),
        ("HostScalar", PyOpenCLHostScalarCompactionAdapter),
        ("HostNonzero", PyOpenCLHostNonzeroCompactionAdapter),
        ("HostCompress", PyOpenCLHostCompressCompactionAdapter),
        ("GPUScalar", PyOpenCLGPUScalarCompactionAdapter),
        ("GPUVectorized", PyOpenCLGPUVectorizedCompactionAdapter),
    ]
    instances = {}
    for name, cls in classes:
        try:
            instances[name] = cls()
        except Exception as exc:
            pytest.skip(f"OpenCL device unavailable for {cls.__name__}: {exc}")
    return instances


def _invasion(shape=(32, 32)) -> GeoTiffMap:
    rng = np.random.default_rng(42)
    # Generate random floats in [0, 1]
    data = rng.random(shape).astype(np.float32)
    return GeoTiffMap(code="invasion", kind="invasion", data=data)


def _habitat(code="H", shape=(32, 32), density=0.5) -> GeoTiffMap:
    rng = np.random.default_rng(100)
    data = (rng.random(shape) < density).astype(np.uint8)
    return GeoTiffMap(code=code, kind="habitat", data=data)


def test_compaction_correctness_across_variants(adapters):
    invasion = _invasion()
    habitat = _habitat()

    results = {}
    for name, adapter in adapters.items():
        (res,) = adapter.batch_preprocess(invasion, [habitat])
        results[name] = res

    baseline_res = results["Baseline"]
    assert baseline_res.n_cells > 0

    for name, res in results.items():
        assert res.habitat_code == "H", f"{name} habitat_code mismatch"
        assert res.n_cells == baseline_res.n_cells, f"{name} n_cells mismatch"
        np.testing.assert_array_almost_equal(
            res.p_vec,
            baseline_res.p_vec,
            err_msg=f"{name} p_vec values mismatch baseline"
        )


def test_empty_habitat_compaction(adapters):
    invasion = _invasion()
    h_data = np.zeros((32, 32), dtype=np.uint8)
    habitat = GeoTiffMap(code="empty", kind="habitat", data=h_data)

    for name, adapter in adapters.items():
        (res,) = adapter.batch_preprocess(invasion, [habitat])
        assert res.n_cells == 0, f"{name} expected 0 cells"
        assert len(res.p_vec) == 0, f"{name} expected empty p_vec"
