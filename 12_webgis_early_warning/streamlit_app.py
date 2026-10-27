"""Interactive dashboard:  streamlit run streamlit_app.py"""
import streamlit as st
from streamlit_folium import st_folium

from webgis_early_warning import alert_grid, build_map, quakes_week, TXT

st.set_page_config(page_title="Pakistan Multi-Hazard EWS (prototype)", layout="wide")
st.title("Pakistan multi-hazard early-warning WebGIS (research prototype)")
st.caption("Layers from snapshot results in data/layers + live USGS / Open-Meteo feeds. Not an official warning system.")

live = st.sidebar.checkbox("Load live feeds", value=True)
m, added = build_map(live=live)
st_folium(m, height=700, use_container_width=True)

if live:
    c1, c2 = st.columns(2)
    try:
        g = alert_grid()
        c1.subheader("Highest rainfall alerts (7-day forecast)")
        c1.dataframe(g.sort_values("rain_alert", ascending=False).head(10).assign(level=lambda d: d.rain_alert.map(TXT)))
        c2.subheader("Earthquakes M2.5+ (7 days)")
        c2.dataframe(quakes_week().sort_values("mag", ascending=False).head(10))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Live tables unavailable: {e}")
