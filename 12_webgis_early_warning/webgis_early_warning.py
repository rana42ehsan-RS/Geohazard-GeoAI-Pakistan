"""12 - WebGIS multi-hazard early warning system for disaster-risk reduction (Pakistan).

Builds outputs/a12/index.html (Leaflet via folium; publish on GitHub Pages / any static host) from snapshot copies of
results stored in data/layers (this folder reads no other folder) plus live feeds: USGS earthquakes (7 days), Open-Meteo rain/heat forecast alert grid, optional NASA FIRMS.
Interactive version:  streamlit run streamlit_app.py
Run:  python webgis_early_warning.py        (layers whose file is missing from data/layers are skipped)
"""
import os
import json

import folium
import numpy as np
import pandas as pd
import requests
from folium import plugins

from common import boundaries, config as C, ml, viz, webdata
from common.report import Report

TAG = "a12"
RASTERS = [  # (output path, layer name, colormap, vmin, vmax)
    ("Flood_susceptibility.tif", "Flood susceptibility (Lower Indus)", "Blues", 0, 1),
    ("Landslide_susceptibility.tif", "Landslide susceptibility (Muzaffarabad-Balakot)", "YlOrBr", 0, 1),
    ("hazard_index_Combined.tif", "MCDA geohazard index (Hazara arc)", "OrRd", 0, 1),
    ("heat_risk.tif", "Urban heat risk (Lahore)", "Reds", 0, 1),
    ("Forest_fire_susceptibility.tif", "Forest-fire susceptibility (Margalla-Galiyat)", "YlOrRd", 0, 1),
]
VECTORS = [  # (geojson, layer, column, caption)
    ("district_risk.geojson", "Multi-hazard risk (districts)", "risk", ["ADM1_NAME", "ADM2_NAME", "hazard", "exposure", "vulnerability", "risk"]),
    ("district_seismic_risk.geojson", "Seismic risk (districts)", "risk", ["ADM1_NAME", "ADM2_NAME", "P50_M6", "vulnerability", "risk"]),
    ("district_cdi_latest.geojson", "Drought index (Balochistan, latest)", "cdi", ["district", "cdi"]),
]


def alert_grid(step=2.0):
    bb = C.SITES["a12"]["bbox"]
    lons = np.arange(bb[0] + 0.5, bb[2], step)
    lats = np.arange(bb[1] + 0.5, bb[3], step)
    LL, LA = np.meshgrid(lons, lats)
    fc = webdata.openmeteo_forecast(LA.ravel(), LL.ravel(), 7)
    rows = []
    for (la, lo), g in fc.groupby(["lat", "lon"]):
        r3 = g["precipitation_sum"].rolling(3).sum().max()
        tmax = g["temperature_2m_max"].max()
        rain = 3 if r3 >= 100 else 2 if r3 >= 60 else 1 if r3 >= 30 else 0
        heat = 3 if tmax >= 47 else 2 if tmax >= 44 else 1 if tmax >= 40 else 0
        rows.append(dict(lat=la, lon=lo, rain3d=round(float(r3), 1), tmax=round(float(tmax), 1), rain_alert=rain, heat_alert=heat))
    return pd.DataFrame(rows)


COL = {0: "#2ca25f", 1: "#fec44f", 2: "#fe9929", 3: "#d7191c"}
TXT = {0: "None", 1: "Watch", 2: "Warning", 3: "Severe"}


def quakes_week():
    j = requests.get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson", timeout=60).json()
    bb = C.SITES["a12"]["bbox"]
    out = []
    for f in j["features"]:
        lon, lat, dep = f["geometry"]["coordinates"]
        if bb[0] <= lon <= bb[2] and bb[1] <= lat <= bb[3]:
            p = f["properties"]
            out.append(dict(lon=lon, lat=lat, depth=dep, mag=p["mag"], place=p["place"], time=pd.to_datetime(p["time"], unit="ms")))
    return pd.DataFrame(out)


def build_map(live=True):
    bb = C.SITES["a12"]["bbox"]
    m = folium.Map(location=[30.5, 69.3], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)      # keyless basemap
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite", show=False).add_to(m)
    added = []

    # reference layers: district outlines (167 districts) and the labelled Kashmir polygon
    dg = boundaries.districts()[["name", "admin1Name", "geometry"]]
    folium.GeoJson(dg.__geo_interface__, name="District boundaries", show=True,
                   style_function=lambda x: {"fillOpacity": 0, "color": "#555555", "weight": 0.6},
                   tooltip=folium.GeoJsonTooltip(fields=["name", "admin1Name"], aliases=["District", "Province"])).add_to(m)
    kg = boundaries.kashmir()
    fk = folium.FeatureGroup(name="Indian Illegally Occupied Jammu & Kashmir", show=True)
    folium.GeoJson(kg.__geo_interface__, style_function=lambda x: {"fillColor": "#e6dccb", "color": "#8c2d04", "weight": 1.5,
                                                                    "fillOpacity": 0.6, "dashArray": "6 4"},
                   tooltip="Indian Illegally Occupied Jammu & Kashmir").add_to(fk)
    pt = kg.geometry.iloc[0].representative_point()
    folium.Marker([pt.y, pt.x], icon=folium.DivIcon(html=(
        '<div style="font-size:11px;font-weight:bold;color:#7f0000;text-align:center;width:190px;margin-left:-95px;'
        'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff">Indian Illegally Occupied<br>Jammu &amp; Kashmir</div>'))).add_to(fk)
    fk.add_to(m)
    added += ["District boundaries", "Kashmir (labelled)"]

    for path, name, col, cap in VECTORS:
        p = C.DATA / "layers" / path
        if not p.exists():
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        data = pd.DataFrame([f["properties"] for f in g["features"]])[["ADM2_CODE", col]].dropna()
        ch = folium.Choropleth(geo_data=g, data=data, columns=["ADM2_CODE", col], key_on="feature.properties.ADM2_CODE",
                               fill_color="RdYlBu" if col == "cdi" else "YlOrRd", fill_opacity=0.7, line_opacity=0.2,
                               name=name, show=False)
        for key in [k for k in ch._children if k.startswith("color_map")]:
            del ch._children[key]            # the stacked colour bars overlap each other; the tooltip layer gives values
        ch.add_to(m)
        folium.GeoJson(g, name=name + " (info)", show=False, style_function=lambda x: {"fillOpacity": 0, "weight": 0},
                       tooltip=folium.GeoJsonTooltip(fields=[c for c in cap if c in g["features"][0]["properties"]])).add_to(m)
        added.append(name)

    for path, name, cmap, lo, hi in RASTERS:
        p = C.DATA / "layers" / path
        if not p.exists():
            continue
        import rasterio
        with rasterio.open(p) as ds:
            a = ds.read(1)
            b = ds.bounds
        rgba = viz.raster_to_rgba(a, cmap, lo, hi)
        folium.raster_layers.ImageOverlay(image=rgba, bounds=[[b.bottom, b.left], [b.top, b.right]], name=name,
                                          opacity=0.85, show=False).add_to(m)
        added.append(name)

    if live:
        try:
            grid = alert_grid()
            grid.to_csv(C.out_dir(TAG) / "alert_grid_latest.csv", index=False)
            fr = folium.FeatureGroup(name="Rainfall alerts (7-day forecast)", show=True)
            fh = folium.FeatureGroup(name="Heat alerts (7-day forecast)", show=False)
            for _, r in grid.iterrows():
                folium.CircleMarker([r.lat, r.lon], radius=7 + 3 * r.rain_alert, color=COL[r.rain_alert], fill=True, fill_opacity=0.8,
                                    tooltip=f"3-day rain peak {r.rain3d} mm -> {TXT[r.rain_alert]}").add_to(fr)
                folium.CircleMarker([r.lat, r.lon], radius=7 + 3 * r.heat_alert, color=COL[r.heat_alert], fill=True, fill_opacity=0.8,
                                    tooltip=f"Tmax {r.tmax} °C -> {TXT[r.heat_alert]}").add_to(fh)
            fr.add_to(m); fh.add_to(m)
            added += ["Rainfall alerts", "Heat alerts"]
        except Exception as e:  # noqa: BLE001
            print("[warn] forecast layer skipped:", e)
        try:
            q = quakes_week()
            fq = folium.FeatureGroup(name="Earthquakes M2.5+ (last 7 days, USGS)", show=True)
            for _, e in q.iterrows():
                folium.CircleMarker([e.lat, e.lon], radius=2 + 2 * e.mag, color="#6a3d9a", fill=True, fill_opacity=0.6,
                                    tooltip=f"M{e.mag} {e.place}<br>{e.time:%Y-%m-%d %H:%M} UTC, depth {e.depth:.0f} km").add_to(fq)
            fq.add_to(m); added.append("Earthquakes")
        except Exception as e:  # noqa: BLE001
            print("[warn] earthquake layer skipped:", e)
        key = os.getenv("FIRMS_KEY")
        if key:
            try:
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bb[0]},{bb[1]},{bb[2]},{bb[3]}/2"
                f = pd.read_csv(url)
                ff = folium.FeatureGroup(name="Active fires (VIIRS, 48 h)", show=True)
                for _, r in f.iterrows():
                    folium.CircleMarker([r.latitude, r.longitude], radius=3, color="#e31a1c", fill=True).add_to(ff)
                ff.add_to(m); added.append("Active fires")
            except Exception as e:  # noqa: BLE001
                print("[warn] FIRMS layer skipped:", e)

    legend = """<div style="position:fixed;bottom:25px;left:10px;z-index:9999;background:white;padding:8px 10px;
    border:1px solid #999;font-size:12px;border-radius:4px"><b>Alert level</b><br>
    <span style="color:#2ca25f">●</span> None &nbsp;<span style="color:#fec44f">●</span> Watch &nbsp;
    <span style="color:#fe9929">●</span> Warning &nbsp;<span style="color:#d7191c">●</span> Severe<br>
    <span style="font-size:10px">Rain: 3-day total ≥30/60/100 mm. Heat: Tmax ≥40/44/47 °C. Research prototype - not an official warning.</span></div>"""
    m.get_root().html.add_child(folium.Element(legend))
    plugins.Fullscreen().add_to(m); plugins.MiniMap(toggle_display=True).add_to(m); plugins.MeasureControl().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m, added


def main():
    C.ensure()
    out = C.out_dir(TAG)
    m, added = build_map(live=True)
    m.save(out / "index.html")
    cfg = C.SITES[TAG]
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. System design")
    r.bullets(["Data layer: snapshot copies of model results (GeoTIFF/GeoJSON in data/layers) and live APIs (USGS ComCat feed, Open-Meteo forecast, optional NASA FIRMS).",
               "Processing layer: this Python script, which can be scheduled (Windows Task Scheduler / cron / GitHub Actions) to refresh the live feeds.",
               "Presentation layer: Leaflet map (folium) with layer control, tooltips, fullscreen, measuring tools; Streamlit dashboard for interactivity.",
               "Alert logic: rule-based thresholds on 3-day forecast rainfall and maximum temperature; earthquake events are shown as published by USGS.",
               "Layers included in this build: " + (", ".join(added) if added else "none (data/layers is empty)")])
    r.h("3. Alert thresholds"); r.table(pd.DataFrame({"Level": ["Watch", "Warning", "Severe"], "3-day rain (mm)": [30, 60, 100], "Tmax (°C)": [40, 44, 47]}), "Prototype thresholds")
    r.h("4. Limitations")
    r.p("Thresholds are generic prototypes; PMD/NDMA operational values vary by region and season. Forecast grid spacing (2°) is coarse. "
        "The system is a decision-support prototype and must not be used as an official warning channel.")
    r.references(["gee"])
    r.save(out / "A12_report.docx")
    print("Done ->", out / "index.html")


if __name__ == "__main__":
    main()
