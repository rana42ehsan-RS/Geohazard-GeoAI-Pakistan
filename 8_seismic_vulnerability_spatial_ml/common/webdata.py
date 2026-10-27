"""Free web APIs: USGS earthquakes, Open-Meteo (archive, forecast, flood) and landslide inventory loader."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def usgs_catalog(bbox, start="1973-01-01", end="2025-12-31", minmag=4.0, step_years=4):
    """Download the USGS ComCat catalogue in time windows (API limit 20 000 events per call)."""
    cache = C.DATA / "tables" / f"usgs_{minmag}_{start[:4]}_{end[:4]}_{'_'.join(map(str, map(int, bbox)))}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["time"])
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1, step_years):
        a, b = f"{y}-01-01", f"{min(y + step_years - 1, int(end[:4]))}-12-31"
        r = requests.get(USGS, params=dict(format="csv", starttime=a, endtime=b, minmagnitude=minmag,
                                           minlatitude=bbox[1], maxlatitude=bbox[3], minlongitude=bbox[0],
                                           maxlongitude=bbox[2], orderby="time-asc", limit=20000), timeout=120)
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text)))
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df[["time", "latitude", "longitude", "depth", "mag", "magType", "place"]].drop_duplicates()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def openmeteo_archive(lat, lon, start, end, hourly=("precipitation",), daily=None):
    p = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, timezone="UTC")
    if hourly:
        p["hourly"] = ",".join(hourly)
    if daily:
        p["daily"] = ",".join(daily)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=120)
    r.raise_for_status()
    j = r.json()
    key = "hourly" if hourly else "daily"
    df = pd.DataFrame(j[key])
    df["time"] = pd.to_datetime(df["time"])
    return df


def openmeteo_forecast(lats, lons, days=7):
    """Daily precipitation forecast for many points (chunks of 50)."""
    out = []
    lats, lons = list(lats), list(lons)
    for i in range(0, len(lats), 50):
        la, lo = lats[i:i + 50], lons[i:i + 50]
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params=dict(latitude=",".join(map(str, la)), longitude=",".join(map(str, lo)),
                                     daily="precipitation_sum,temperature_2m_max", forecast_days=days,
                                     timezone="UTC"), timeout=60)
        r.raise_for_status()
        j = r.json()
        j = j if isinstance(j, list) else [j]
        for k, rec in enumerate(j):
            d = pd.DataFrame(rec["daily"])
            d["lat"], d["lon"] = la[k], lo[k]
            out.append(d)
    return pd.concat(out, ignore_index=True)


def openmeteo_discharge(lat, lon, start, end):
    """GloFAS v4 modelled river discharge (m3/s), daily. Modelled, not observed - state this in the report."""
    r = requests.get("https://flood-api.open-meteo.com/v1/flood",
                     params=dict(latitude=lat, longitude=lon, daily="river_discharge",
                                 start_date=start, end_date=end), timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"river_discharge": "q_glofas"})


def load_landslide_inventory(bbox):
    """Return (lon, lat) array of landslide points inside bbox, or None.

    Priority: your own file in data/inventories (csv with lon/lat or any vector file) ->
    NASA Global Landslide Catalog download. Returns None if neither is available.
    """
    import geopandas as gpd
    inv = C.DATA / "inventories"
    for f in list(inv.glob("*.shp")) + list(inv.glob("*.gpkg")) + list(inv.glob("*.geojson")):
        g = gpd.read_file(f)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if (g.geometry.geom_type != "Point").any():
            g = g.set_geometry(g.geometry.centroid)        # polygon inventory -> one point per mapped failure (native CRS)
        g = g.to_crs(4326)
        xy = np.c_[g.geometry.x, g.geometry.y]
        return _inside(xy, bbox)
    for f in inv.glob("*.csv"):
        d = pd.read_csv(f)
        lo = next((c for c in d.columns if c.lower() in ("lon", "longitude", "x")), None)
        la = next((c for c in d.columns if c.lower() in ("lat", "latitude", "y")), None)
        if lo and la:
            return _inside(d[[lo, la]].to_numpy(float), bbox)
    try:  # NASA COOLR / Global Landslide Catalog
        url = "https://data.nasa.gov/api/views/dd9e-wu2v/rows.csv?accessType=DOWNLOAD"
        d = pd.read_csv(url)
        cache = inv / "nasa_global_landslide_catalog.csv"
        d.to_csv(cache, index=False)
        xy = d[["longitude", "latitude"]].dropna().to_numpy(float)
        return _inside(xy, bbox)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Could not download NASA landslide catalogue ({e}).")
        return None


def _inside(xy, bbox):
    m = (xy[:, 0] >= bbox[0]) & (xy[:, 0] <= bbox[2]) & (xy[:, 1] >= bbox[1]) & (xy[:, 1] <= bbox[3])
    return xy[m]


def inventory_extent(xy, pad=0.01):
    """Bounding box (west, south, east, north) of the mapped area covered by an inventory, padded slightly."""
    return (xy[:, 0].min() - pad, xy[:, 1].min() - pad, xy[:, 0].max() + pad, xy[:, 1].max() + pad)
