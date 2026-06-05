"""Synthetic habitat and invasion map generation based on DEM.

Provides functions to create physically coherent synthetic habitats and invasion
probability maps using a Digital Elevation Model (DEM).
"""

from __future__ import annotations
import numpy as np

from pyroclast.ABCs.repository import MapCriteria, RasterMap


class MemoryRasterMap(RasterMap):
    """An in-memory implementation of RasterMap for synthetic data."""

    def __init__(self, code: str, kind: str, data: np.ndarray) -> None:
        self._code = code
        self._kind = kind
        self._data = data

    @property
    def code(self) -> str:
        return self._code

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def data(self) -> np.ndarray:
        return self._data

    def satisfies(self, criteria: MapCriteria) -> bool:
        from pyroclast.io.data_repository import HabitatCriteria, InvasionCriteria
        if isinstance(criteria, InvasionCriteria):
            return self._kind == "invasion"
        if isinstance(criteria, HabitatCriteria):
            return self._kind == "habitat" and (
                criteria.code is None or criteria.code == self._code
            )
        return False


def generate_synthetic_habitat_dem(
    dem: np.ndarray,
    occupancy_fraction: float,
    mean_p: float,
    seed: int = 42,
    habitat_code: str = "SYNTH_DEM",
    threshold: float = 0.28,
) -> tuple[RasterMap, RasterMap]:
    """Generate a synthetic habitat map and an invasion probability map based on a DEM.

    Parameters
    ----------
    dem : np.ndarray
        2-D array representing the Digital Elevation Model.
        Invalid cells should be represented by NaN or values <= -9999.
    occupancy_fraction : float
        The desired fraction of valid land cells to be occupied by the habitat [0.0, 1.0].
    mean_p : float
        The target average invasion probability over all valid land cells [0.0, 1.0].
    seed : int, optional
        NumPy random seed for reproducibility. Default: 42.
    habitat_code : str, optional
        The code for the generated habitat. Default: "SYNTH_DEM".
    threshold : float, optional
        The critical fraction threshold theta for the generated habitat. Default: 0.28.

    Returns
    -------
    habitat_map : RasterMap
        A synthetic habitat RasterMap (kind="habitat", uint8 presence mask).
    invasion_map : RasterMap
        A synthetic invasion probability RasterMap (kind="invasion", float32).
    """
    if not (0.0 <= occupancy_fraction <= 1.0):
        raise ValueError("occupancy_fraction must be in [0.0, 1.0]")
    if not (0.0 <= mean_p <= 1.0):
        raise ValueError("mean_p must be in [0.0, 1.0]")

    # 1. Identify valid land cells
    valid_mask = ~np.isnan(dem) & (dem > -9999.0)
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise ValueError("DEM contains no valid land cells")

    valid_z = dem[valid_mask]
    rng = np.random.default_rng(seed)

    # 2. Generate habitat footprint (presence_mask) using a random altitudinal band
    presence_mask = np.zeros(dem.shape, dtype=np.uint8)
    if occupancy_fraction > 0.0:
        if occupancy_fraction >= 1.0:
            presence_mask[valid_mask] = 1
        else:
            # Choose a random percentile interval of length occupancy_fraction
            p_low = rng.uniform(0.0, 1.0 - occupancy_fraction)
            p_high = p_low + occupancy_fraction

            z_low = np.percentile(valid_z, p_low * 100.0)
            z_high = np.percentile(valid_z, p_high * 100.0)

            # Mark active cells within this band as occupied
            presence_mask[valid_mask & (dem >= z_low) & (dem <= z_high)] = 1

    # 3. Generate invasion probability map (p_map)
    p_map_data = np.zeros(dem.shape, dtype=np.float32)
    if mean_p > 0.0:
        if mean_p >= 1.0:
            p_map_data[valid_mask] = 1.0
        else:
            # Generate uniform random noise centered around mean_p
            # Restrict noise range dynamically to keep probabilities strictly in [0, 1] before clipping
            noise_range = min(mean_p, 1.0 - mean_p, 0.15)
            if noise_range > 0.0:
                raw_noise = rng.uniform(-noise_range, noise_range, size=n_valid)
                # Adjust noise to have exactly 0.0 mean, guaranteeing mean(p) == mean_p
                raw_noise -= np.mean(raw_noise)
                p_map_data[valid_mask] = np.clip(mean_p + raw_noise, 0.0, 1.0).astype(np.float32)
            else:
                p_map_data[valid_mask] = mean_p

    habitat_map = MemoryRasterMap(habitat_code, "habitat", presence_mask)
    invasion_map = MemoryRasterMap("invasion", "invasion", p_map_data)

    return habitat_map, invasion_map
