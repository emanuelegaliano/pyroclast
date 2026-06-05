"""Unit and integration tests for the DEM-based synthetic habitat generator."""

import numpy as np
import pytest

from pyroclast import generate_synthetic_habitat_dem
from pyroclast.adapters.opencl_adapter import PyOpenCLAdapter


def test_invalid_arguments():
    dem = np.array([[100, 200], [300, 400]], dtype=np.float32)
    with pytest.raises(ValueError, match="occupancy_fraction"):
        generate_synthetic_habitat_dem(dem, -0.1, 0.5)
    with pytest.raises(ValueError, match="occupancy_fraction"):
        generate_synthetic_habitat_dem(dem, 1.1, 0.5)
    with pytest.raises(ValueError, match="mean_p"):
        generate_synthetic_habitat_dem(dem, 0.5, -0.1)
    with pytest.raises(ValueError, match="mean_p"):
        generate_synthetic_habitat_dem(dem, 0.5, 1.1)


def test_empty_dem():
    # All cells invalid
    dem = np.full((10, 10), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="no valid land cells"):
        generate_synthetic_habitat_dem(dem, 0.5, 0.5)


def test_dem_occupancy_fraction():
    # 2D DEM with 100 valid cells, 4 invalid cells (NaN)
    dem = np.arange(104, dtype=np.float32).reshape(13, 8)
    dem[0, 0] = np.nan
    dem[1, 1] = -9999.0
    dem[2, 2] = np.nan
    dem[3, 3] = -10000.0

    valid_mask = ~np.isnan(dem) & (dem > -9999.0)
    n_valid = int(np.sum(valid_mask))
    assert n_valid == 100

    # Test occupancy: 30%
    hab, _ = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.3, mean_p=0.5, seed=42)
    assert hab.kind == "habitat"
    assert hab.code == "SYNTH_DEM"
    assert hab.data.shape == dem.shape
    assert hab.data.dtype == np.uint8

    # Occupied count must be exactly 30% of valid cells
    n_occupied = int(np.sum(hab.data))
    assert n_occupied == 30

    # Check that invalid cells are never occupied
    assert np.all(hab.data[~valid_mask] == 0)

    # Test occupancy: 0%
    hab_0, _ = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.0, mean_p=0.5)
    assert np.sum(hab_0.data) == 0

    # Test occupancy: 100%
    hab_100, _ = generate_synthetic_habitat_dem(dem, occupancy_fraction=1.0, mean_p=0.5)
    assert np.sum(hab_100.data) == 100
    assert np.all(hab_100.data[valid_mask] == 1)


def test_dem_mean_probability():
    dem = np.arange(100, dtype=np.float32).reshape(10, 10)
    valid_mask = ~np.isnan(dem) & (dem > -9999.0)

    for target_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        _, inv = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.5, mean_p=target_p)
        assert inv.kind == "invasion"
        assert inv.code == "invasion"
        assert inv.data.shape == dem.shape
        assert inv.data.dtype == np.float32

        # Average probability on valid cells must match target
        mean_p = float(np.mean(inv.data[valid_mask]))
        assert mean_p == pytest.approx(target_p, abs=1e-3)

        # Inactive cells must be exactly 0.0
        assert np.all(inv.data[~valid_mask] == 0.0)

    # Test 0.0 and 1.0 probability corner cases
    _, inv_0 = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.5, mean_p=0.0)
    assert np.all(inv_0.data == 0.0)

    _, inv_1 = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.5, mean_p=1.0)
    assert np.all(inv_1.data[valid_mask] == 1.0)
    assert np.all(inv_1.data[~valid_mask] == 0.0)


def test_dem_flat_terrain():
    # If DEM is completely flat, average probability should still match target_p
    dem = np.full((10, 10), 150.0, dtype=np.float32)
    valid_mask = ~np.isnan(dem) & (dem > -9999.0)

    _, inv = generate_synthetic_habitat_dem(dem, occupancy_fraction=0.4, mean_p=0.6)
    assert np.mean(inv.data[valid_mask]) == pytest.approx(0.6)
def test_integration_with_preprocessing_adapter():
    """Verify that generated synthetic maps can be preprocessed by the GPU compute adapter."""
    try:
        adapter = PyOpenCLAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")

    # Generate synthetic maps
    dem = np.arange(100, dtype=np.float32).reshape(10, 10)
    hab_map, inv_map = generate_synthetic_habitat_dem(
        dem, occupancy_fraction=0.4, mean_p=0.5, habitat_code="INTEGRATION_TEST"
    )

    # Pass them directly to the adapter's batch_preprocess
    results = adapter.batch_preprocess(inv_map, [hab_map])

    assert len(results) == 1
    compacted = results[0]
    assert compacted.habitat_code == "INTEGRATION_TEST"
    assert compacted.n_cells == 40  # 40% of 100 cells
    assert compacted.p_vec.shape == (40,)
    assert compacted.p_vec.dtype == np.float32
    assert 0.0 <= compacted.mean_probability <= 1.0
