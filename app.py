import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import glob
from typing import Optional

try:
    import geopandas as gpd  # optional
except Exception:
    gpd = None

from streamlit_folium import st_folium  # type: ignore
import folium  # type: ignore
from folium.plugins import HeatMap, Draw, MeasureControl, Fullscreen  # type: ignore

with st.sidebar:
    map_only = st.toggle("Map-only mode", value=False)
    use_folium = st.toggle("Advanced map (Folium)", value=False, help="Enable drawing, measuring, base layers, and GeoJSON overlays.")

#st.write("Streamlit has lots of fans in the geo community. 🌍 It supports maps from PyDeck, Folium, Kepler.gl, and others.")

chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [22.5431, 114.0579],
   columns=['lat', 'lon'])

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

if use_folium:
    # Folium map with useful controls
    m = folium.Map(location=[22.5431, 114.0579], zoom_start=11, tiles="CartoDB positron", control_scale=True)

    if not map_only:
        # Heat layer from synthetic points
        HeatMap(chart_data[["lat", "lon"]].values.tolist(), radius=25, blur=15, min_opacity=0.2).add_to(m)
        # SZ districts overlay if available
        _add_sz_districts_overlay(m)

        # Helpful controls
        Draw(export=False).add_to(m)
        MeasureControl(position="topleft", primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
        Fullscreen(position="topleft").add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, height=650, returned_objects=[])
else:
    map_layers = [
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ]

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=22.5431,
            longitude=114.0579,
            zoom=11,
            pitch=50,
        ),
        layers=[] if map_only else map_layers,
    ))