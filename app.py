import streamlit as st
import os
import glob
from typing import Optional

try:
    import geopandas as gpd  # optional
except Exception:
    gpd = None

from streamlit_folium import st_folium  # type: ignore
import folium  # type: ignore
from folium.plugins import Draw, MeasureControl, Fullscreen  # type: ignore

st.set_page_config(page_title="Shenzhen EV Map", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        text-align: center;
    }
    iframe[title="st_folium"] {
        margin-left: auto !important;
        margin-right: auto !important;
        display: block;
        aspect-ratio: 16 / 9;
        width: 90vw !important;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h2 style='text-align:center;margin-top:0;'>EV Charger Occupancy Predictor</h2>", unsafe_allow_html=True)

def _find_sz_districts() -> Optional[str]:
    candidates = [
        "data/SZ_districts.geojson",
        "data/SZ_districts.json",
        "data/SZ_districts.gpkg",
        "data/SZ_districts.shp",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    folder = "data/SZ_districts"
    if os.path.isdir(folder):
        for pattern in ("*.geojson", "*.json", "*.gpkg", "*.shp"):
            matches = sorted(glob.glob(os.path.join(folder, pattern)))
            if matches:
                return matches[0]

    return None

def _add_sz_districts_overlay(m: folium.Map) -> None:
    src = _find_sz_districts()
    if not src:
        return
    try:
        if gpd is not None:
            gdf = gpd.read_file(src)  # type: ignore
            if not gdf.empty:
                gdf = gdf.to_crs(4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
                gj = folium.GeoJson(
                    data=gdf.to_json(),
                    name="SZ_districts",
                    style_function=lambda f: {"fillColor": "#3388ff40", "color": "#3388ff", "weight": 1},
                    tooltip=folium.GeoJsonTooltip(fields=[c for c in gdf.columns if c != "geometry"][:8])
                )
                gj.add_to(m)
        else:
            # Fallback: try to pass raw file
            folium.GeoJson(src, name="SZ_districts").add_to(m)
    except Exception:
        pass

# Folium map (single view, no sidebar toggles)
m = folium.Map(location=[22.5431, 114.0579], zoom_start=11, tiles="CartoDB positron", control_scale=True)

# SZ districts overlay if available
_add_sz_districts_overlay(m)

# Helpful controls directly on the map
Draw(export=False).add_to(m)
MeasureControl(position="topleft", primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
Fullscreen(position="topleft").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

_, center_col, _ = st.columns([1, 3, 1])
with center_col:
    st_folium(m, height=720, returned_objects=[])