import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors
import rasterio
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    InvasionCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
)
from pyroclast.domain.models import MonteCarloConfig
from pyroclast.services import run_preprocessing_batch

# ── 1. Page Config & Load Environment ─────────────────────────
st.set_page_config(
    page_title="Pyroclast Demo — Monte Carlo GPU",
    page_icon="🌋",
    layout="wide",
)

load_dotenv()

# ── 2. Constants & Helpers ────────────────────────────────────
def get_habitat_colors(codes):
    """Generate a stable color mapping for a list of habitat codes."""
    cmap = plt.get_cmap("tab10") 
    return {code: matplotlib.colors.to_hex(cmap(i % 10)) for i, code in enumerate(sorted(codes))}

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

# ── 3. Sidebar Configuration ──────────────────────────────────
st.sidebar.title("🌋 Pyroclast Config")

data_path_env = os.getenv("DATA_PATH", "data").strip('"\' ')
data_path = st.sidebar.text_input("Data Path", value=data_path_env)
invasion_map_env = os.getenv("INVASION_MAP", "").strip('"\' ')

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo Parameters")
mc_runs = int(st.sidebar.number_input("Total Simulations (R)", value=1_000_000, step=100_000, min_value=1))
mc_threshold = float(st.sidebar.number_input("Threshold (θ)", value=0.005, format="%.4f", step=0.0001, min_value=0.0, max_value=1.0))
mc_seed = int(st.sidebar.number_input("Random Seed", value=42, help="Il seed inizializza il generatore di numeri casuali MWC64X. Usare lo stesso seed permette di replicare esattamente gli stessi risultati."))
mc_batches = int(st.sidebar.number_input("Batches", value=10, min_value=1))

# Choose Monte Carlo Kernel/Adapter
mc_adapter_choice = st.sidebar.selectbox(
    "Kernel Variant",
    options=[
        "Standard (1-D)",
        "Ping-Pong (1-D)",
    ],
    index=0
)

st.sidebar.markdown("---")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.info("Cache cleared.")

# ── 4. Main App Logic ─────────────────────────────────────────
st.title("Monte Carlo GPU Simulation")
st.markdown(f"""
Questa demo esegue la simulazione Monte Carlo per il calcolo della probabilità di distruzione 
degli habitat a causa della colata lavica, utilizzando il kernel **{mc_adapter_choice}** su GPU.
""")

_MC_CHOICES = {
    "Standard (1-D)": PyOpenCLMonteCarloAdapter,
    "Ping-Pong (1-D)": PyOpenCLMonteCarloPingPongAdapter,
}

@st.cache_resource
def get_preprocess_adapter():
    return PyOpenCLAdapter()

@st.cache_resource
def get_mc_adapter(choice: str):
    adapter_cls = _MC_CHOICES[choice]
    return adapter_cls(profiling=True)

@st.cache_data
def load_data(path, invasion_map_path):
    repo = FileMapRepository(path, invasion_map=invasion_map_path)
    habitats = repo.matching(HabitatCriteria())
    invasion = repo.get(InvasionCriteria())
    return repo, habitats, invasion

try:
    preprocess_adapter = get_preprocess_adapter()
    mc_adapter = get_mc_adapter(mc_adapter_choice)
    repo, habitats, invasion = load_data(data_path, invasion_map_env)
    
    all_codes = [h.code for h in habitats]
    habitat_colors = get_habitat_colors(all_codes)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Habitats Selection")
        selected_codes = st.multiselect(
            "Select Habitats to display and simulate",
            options=all_codes,
            default=[all_codes[0]] if all_codes else []
        )
        
        if not selected_codes:
            st.warning("Please select at least one habitat.")
        else:
            cache_dir = Path(data_path) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with st.spinner("Compacting habitats..."):
                compacted_list = run_preprocessing_batch(
                    repo=repo,
                    compute=preprocess_adapter,
                    criteria=HabitatCriteria(code=None),
                    cache_dir=cache_dir
                )
                compacted_habitats = [h for h in compacted_list if h.habitat_code in selected_codes]
            
            summary_data = []
            for ch in compacted_habitats:
                color = habitat_colors.get(ch.habitat_code, "#FFFFFF")
                summary_data.append({
                    "Color": "●",
                    "Habitat": ch.habitat_code,
                    "At-risk Cells": ch.n_cells,
                    "Mean P": round(ch.mean_probability, 6),
                    "_color": color
                })
            
            df = pd.DataFrame(summary_data)
            def style_df(styler):
                for i, row in df.iterrows():
                    styler.set_properties(subset=pd.IndexSlice[i, "Color"], **{'color': row._color, 'font-size': '20px'})
                return styler

            st.dataframe(
                df.drop(columns=["_color"]).style.pipe(style_df),
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("🚀 Run Monte Carlo for all selected", type="primary"):
                st.subheader("Simulation Results")
                
                results = {}
                for hab in compacted_habitats:
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    config = MonteCarloConfig(
                        n_runs=mc_runs,
                        threshold=mc_threshold,
                        seed=mc_seed
                    )
                    
                    def _progress_callback(i, total, partial_prob, code=hab.habitat_code):
                        progress = (i + 1) / total
                        progress_bar.progress(progress)
                        status_placeholder.text(f"Habitat {code}: Batch {i+1}/{total} — P ≈ {partial_prob:.6f}")
                    
                    prob = mc_adapter.run_batched(
                        hab, 
                        config, 
                        n_batches=mc_batches, 
                        callback=_progress_callback
                    )
                    results[hab.habitat_code] = prob
                    status_placeholder.empty()
                    progress_bar.empty()
                
                st.write("### Final Probabilities")
                for code, p in results.items():
                    color = habitat_colors.get(code, "#FFFFFF")
                    st.markdown(f"<span style='color:{color}; font-size:1.5rem;'>●</span> Habitat **{code}** : <span style='color:{color}; font-weight:bold; font-size:1.2rem;'>{p:.6f}</span>", unsafe_allow_html=True)
                
                if mc_adapter._kernel_launches:
                    bench = mc_adapter.benchmark()[0]
                    st.info(f"**Performance:** {bench.mean_ms:.3f} ms/kernel | {bench.bandwidth_gbs:.2f} GB/s")
                    mc_adapter.reset_profile()

    with col2:
        st.subheader("Map Visualization")
        
        @st.cache_data
        def generate_map(path, invasion_map_path):
            dem_path_env = os.getenv("DEM_PATH", "").strip('"\' ')
            if not dem_path_env:
                raise ValueError("DEM_PATH not set in .env")
            
            dem_path = Path(dem_path_env)
            inv_p = Path(invasion_map_path) if invasion_map_path else sorted(Path(path).glob("*.tif"))[0]
            
            with rasterio.open(inv_p) as ref:
                ref_crs = ref.crs
                ref_transform = ref.transform
                ref_shape = ref.shape
            
            dem = _load_dem_reprojected(dem_path, ref_crs, ref_transform, ref_shape)
            shade = _hillshade(dem)
            return shade, dem
        
        if selected_codes:
            try:
                shade, dem = generate_map(data_path, invasion_map_env)
                
                fig, ax = plt.subplots(figsize=(8, 7))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                
                ax.imshow(shade, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
                
                dem_masked = np.where(np.isnan(dem), np.nanmin(dem), dem)
                ax.imshow(dem_masked, cmap="terrain", alpha=0.3, interpolation="bilinear")
                
                inv_data = invasion.data
                inv_masked = np.ma.masked_where(inv_data == 0, inv_data)
                ax.imshow(inv_masked, cmap="YlOrRd", alpha=0.5)
                
                legend_patches = []
                for code in selected_codes:
                    h_map = next(h for h in habitats if h.code == code)
                    mask = h_map.data
                    color = habitat_colors.get(code, "#00ff00")
                    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                    r, g, b = matplotlib.colors.to_rgb(color)
                    rgba[mask == 1] = [r, g, b, 0.7]
                    ax.imshow(rgba, interpolation="none")
                    legend_patches.append(mpatches.Patch(color=color, label=f"Habitat {code}"))
                
                if legend_patches:
                    ax.legend(handles=legend_patches, loc="lower left", facecolor="#1e1e1e", labelcolor="white")
                
                ax.axis("off")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating map: {e}")
                st.warning("Ensure DEM_PATH is correct in .env and files exist.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check your .env configuration and data path.")
