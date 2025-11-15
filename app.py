import copy
import glob
import json
import os
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import geopandas as gpd  
except Exception:
    gpd = None

import folium 
import pydeck as pdk  
from branca.colormap import linear
from folium.plugins import Draw, MeasureControl, Fullscreen  
from streamlit_folium import st_folium  

st.set_page_config(page_title="Shenzhen EV Map", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        text-align: center;
    }
    .main .block-container {
        max-width: 100%;
        width: 100%;
        padding-left: 0;
        padding-right: 0;
    }
    .map-wrapper {
        display: flex;
        justify-content: center;
        width: 100%;
        overflow-x: auto;
    }
    .map-wrapper iframe[title="st_folium"] {
        width: min(100vw, 1900px) !important;
        aspect-ratio: 16 / 9;
        min-width: 900px;
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


@st.cache_data(show_spinner=False)
def load_district_geojson() -> Optional[dict]:
    if gpd is None:
        return None
    src = _find_sz_districts()
    if not src:
        return None
    try:
        gdf = gpd.read_file(src)  # type: ignore
        if gdf.empty:
            return None
        gdf = gdf.to_crs(4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
        return json.loads(gdf.to_json())
    except Exception:
        return None


def _prepare_geojson_with_values(
    base_geojson: Optional[dict],
    value_lookup: dict[int, float],
    color_scale,
) -> Optional[dict]:
    if not base_geojson:
        return None
    prepared = copy.deepcopy(base_geojson)
    for feature in prepared.get("features", []):
        props = feature.setdefault("properties", {})
        zone_raw = props.get("TAZID") or props.get("ZONE")
        try:
            zone_id = int(zone_raw) if zone_raw is not None else None
        except (TypeError, ValueError):
            zone_id = None
        value = value_lookup.get(zone_id) if zone_id is not None else None
        if value is not None and not pd.isna(value):
            hex_color = color_scale(value)
            rgba = _hex_to_rgba(hex_color, alpha=160)
        else:
            hex_color = "#d0d2d6"
            rgba = _hex_to_rgba(hex_color, alpha=80)
        props["zone_id"] = zone_id
        props["occupancy"] = None if value is None else float(value)
        props["fill_color_hex"] = hex_color
        props["fill_color_rgba"] = rgba
    return prepared


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
    zone_value_lookup: dict[int, float] = {}
else:
    max_occupancy = max(map_points["occupancy"].max(), 1.0)
    zone_value_lookup = dict(
        zip(map_points["zone_id"].astype(int), map_points["occupancy"].astype(float))
    )
color_scale = linear.YlOrRd_09.scale(0, max_occupancy)


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]

smooth_mode = st.toggle(
    "Smooth WebGL map (beta)",
    value=True,
    help="Uses a WebGL map for faster color updates. Disable to access Folium drawing controls.",
)
base_geojson = load_district_geojson()
choropleth_geojson = _prepare_geojson_with_values(
    base_geojson, zone_value_lookup, color_scale
)

def _add_sz_districts_overlay(m: folium.Map, geojson_data: Optional[dict]) -> None:
    if not geojson_data:
        src = _find_sz_districts()
        if not src:
            return
        folium.GeoJson(src, name="SZ_districts").add_to(m)
        return

    def style_fn(feature):
        return {
            "fillColor": feature["properties"].get("fill_color_hex", "#cccccc"),
            "color": "#1E3A5F",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["zone_id", "occupancy"],
        aliases=["Zone", "Occupancy"],
        localize=True,
        sticky=False,
    )
    folium.GeoJson(
        data=geojson_data,
        style_function=style_fn,
        highlight_function=lambda feat: {
            "weight": 2,
            "color": "#ff9800",
        },
        tooltip=tooltip,
        name="SZ districts",
    ).add_to(m)


def _render_pydeck_map(district_geojson: Optional[dict]) -> None:
    if not district_geojson or not district_geojson.get("features"):
        st.warning("District polygons unavailable for smooth rendering.")
        return

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            data=district_geojson,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color_rgba",
            get_line_color=[40, 40, 40, 120],
            line_width_min_pixels=1.2,
            pickable=True,
        )
    ]

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=22.55,
            longitude=114.05,
            zoom=10,
            min_zoom=8,
            max_zoom=13,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": "Zone {properties.zone_id}\nOccupancy: {properties.occupancy}"},
    )
    st.pydeck_chart(deck, use_container_width=True, height=720)

if smooth_mode:
    _render_pydeck_map(choropleth_geojson)
else:
    # Folium map (single view, no sidebar toggles)
    m = folium.Map(
        location=[22.5431, 114.0579],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )
    if not metadata_df.empty:
        lat_min = float(metadata_df["latitude"].min())
        lat_max = float(metadata_df["latitude"].max())
        lon_min = float(metadata_df["longitude"].min())
        lon_max = float(metadata_df["longitude"].max())
        padding = 0.05
        bounds = [
            [lat_min - padding, lon_min - padding],
            [lat_max + padding, lon_max + padding],
        ]
        m.fit_bounds(bounds)

    # SZ districts overlay if available
    _add_sz_districts_overlay(m, choropleth_geojson)

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
    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        primary_area_unit="sqmeters",
    ).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    with st.container():
        st.markdown('<div class="map-wrapper">', unsafe_allow_html=True)
        st_folium(m, height=720, width=None, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)