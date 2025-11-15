import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

with st.sidebar:
    map_only = st.toggle("Map-only mode", value=False)

#st.write("Streamlit has lots of fans in the geo community. 🌍 It supports maps from PyDeck, Folium, Kepler.gl, and others.")

chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [22.5431, 114.0579],
   columns=['lat', 'lon'])

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