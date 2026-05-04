"""Standalone visualization: DEM hillshade with habitat overlays."""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

load_dotenv()

HABITAT_COLORS = {
    "4090":  "#e41a1c",
    "8320":  "#377eb8",
    "92XX":  "#4daf4a",
    "9340":  "#ff7f00",
    "9530_": "#984ea3",
}

_HABITAT_RE = re.compile(r"^cb_codice_(.+)\.tif$")

SAVE_FIGURE = False


def _hillshade(elevation: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    azimuth = np.radians(360.0 - azimuth_deg)
    altitude = np.radians(altitude_deg)
    dy, dx = np.gradient(elevation.astype(np.float64))
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    shade = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def _load_dem_reprojected(dem_path: Path, ref_crs, ref_transform, ref_shape) -> np.ndarray:
    rows, cols = ref_shape
    dst = np.empty((rows, cols), dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_crs=src.crs,
            src_transform=src.transform,
            dst_crs=ref_crs,
            dst_transform=ref_transform,
            resampling=Resampling.bilinear,
        )
    dst[dst <= -9999] = np.nan
    return dst


def main() -> None:
    invasion_path = Path(os.environ["INVASION_MAP"])
    dem_path = Path(os.environ["DEM_PATH"])
    habitats_dir = Path(os.environ["HABITATS_DIR"])

    if not dem_path.is_file():
        raise FileNotFoundError(
            f"DEM file not found: {dem_path}\n"
            "Extract the DEM archive and update DEM_PATH in .env."
        )

    with rasterio.open(invasion_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = ref.shape

    print("Caricamento DEM e ricampionamento a 20m...")
    dem = _load_dem_reprojected(dem_path, ref_crs, ref_transform, ref_shape)

    print("Calcolo hillshade...")
    shade = _hillshade(dem)

    print("Caricamento habitat...")
    habitats: dict[str, np.ndarray] = {}
    for path in sorted(habitats_dir.glob("cb_codice_*.tif")):
        m = _HABITAT_RE.match(path.name)
        if not m:
            continue
        code = m.group(1)
        with rasterio.open(path) as src:
            habitats[code] = src.read(1).astype(np.uint8)
        print(f"  habitat {code}: {int(np.sum(habitats[code])):,} celle")

    print("Rendering...")
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.imshow(shade, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    dem_masked = np.where(np.isnan(dem), np.nanmin(dem), dem)
    terrain_img = ax.imshow(
        dem_masked, cmap="terrain", alpha=0.4, interpolation="bilinear"
    )

    legend_patches = []
    for code, mask in habitats.items():
        color = HABITAT_COLORS.get(code, "#aaaaaa")
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        r, g, b = tuple(int(color.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        rgba[mask == 1] = [r, g, b, 0.55]
        ax.imshow(rgba, interpolation="none")
        legend_patches.append(mpatches.Patch(color=color, label=f"Habitat {code}"))

    cbar = fig.colorbar(terrain_img, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Elevazione (m)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        framealpha=0.7,
        fontsize=10,
        facecolor="#1a1a2e",
        labelcolor="white",
        edgecolor="#555555",
    )

    ax.set_title("DEM + Habitat — Area Etna", color="white", fontsize=14, pad=12)
    ax.axis("off")

    if SAVE_FIGURE:
        out_path = Path(__file__).parent / "visualize_output.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Salvato: {out_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
