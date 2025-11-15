import streamlit as st
import pandas as pd
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
from branca.colormap import linear

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

@st.cache_data(show_spinner=False)
def load_station_metadata() -> pd.DataFrame:
    df = pd.read_csv("data/information.csv")
    df = df.rename(columns={"grid": "zone_id", "lon": "longitude", "la": "latitude"})
    keep_cols = ["zone_id", "longitude", "latitude", "count", "CBD", "dynamic_pricing"]
    return df[keep_cols]


@st.cache_data(show_spinner=False)
def load_occupancy() -> pd.DataFrame:
    occ = pd.read_csv("data/occupancy.csv")
    occ.columns = occ.columns.astype(str)
    occ["timestamp"] = occ["timestamp"].astype(int)
    return occ


@st.cache_data(show_spinner=False)
def load_time_lookup() -> pd.DataFrame:
    time_df = pd.read_csv("data/time.csv")
    time_df["timestamp"] = range(1, len(time_df) + 1)
    dt = pd.to_datetime(
        dict(
            year=time_df["year"],
            month=time_df["month"],
            day=time_df["day"],
            hour=time_df["hour"],
            minute=time_df["minute"],
            second=time_df["second"],
        )
    )
    time_df["label"] = dt.dt.strftime("%Y-%m-%d %H:%M")
    return time_df[["timestamp", "label"]]


metadata_df = load_station_metadata()
occupancy_df = load_occupancy()
time_lookup_df = load_time_lookup()

min_timestamp = int(occupancy_df["timestamp"].min())
max_timestamp = int(occupancy_df["timestamp"].max())

selected_timestamp = st.slider(
    "Select time index",
    min_value=min_timestamp,
    max_value=max_timestamp,
    value=min_timestamp,
    step=1,
)

label_row = time_lookup_df.loc[time_lookup_df["timestamp"] == selected_timestamp]
selected_label = (
    label_row["label"].iloc[0] if not label_row.empty else f"Index {selected_timestamp}"
)
st.caption(f"Showing occupancy at {selected_label} (index {selected_timestamp})")

snapshot = occupancy_df[occupancy_df["timestamp"] == selected_timestamp]
if snapshot.empty:
    st.error("No occupancy data for the selected timestamp.")
    st.stop()

row = snapshot.iloc[0].drop("timestamp")
snapshot_df = (
    row.to_frame(name="occupancy")
    .reset_index()
    .rename(columns={"index": "zone_id"})
)
snapshot_df["zone_id"] = snapshot_df["zone_id"].astype(int)
snapshot_df["occupancy"] = snapshot_df["occupancy"].astype(float)

map_points = metadata_df.merge(snapshot_df, on="zone_id", how="inner")
if map_points.empty:
    max_occupancy = 1.0
else:
    max_occupancy = max(map_points["occupancy"].max(), 1.0)
color_scale = linear.YlOrRd_09.scale(0, max_occupancy)

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

# Occupancy markers
if map_points.empty:
    st.warning("No station metadata matched the occupancy columns.")
else:
    for _, point in map_points.iterrows():
        value = point["occupancy"]
        color = color_scale(value)
        radius = 4 + (value / max_occupancy) * 8
        folium.CircleMarker(
            location=[point["latitude"], point["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(
                f"Zone {point['zone_id']}<br>Occupancy: {value:.0f}<br>Chargers: {point['count']}",
                max_width=200,
            ),
        ).add_to(m)
    color_scale.caption = "Occupied chargers"
    color_scale.add_to(m)

# Helpful controls directly on the map
Draw(export=False).add_to(m)
MeasureControl(position="topleft", primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
Fullscreen(position="topleft").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

_, center_col, _ = st.columns([1, 3, 1])
with center_col:
    st_folium(m, height=720, returned_objects=[])