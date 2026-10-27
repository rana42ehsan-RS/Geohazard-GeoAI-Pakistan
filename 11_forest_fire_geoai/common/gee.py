"""Google Earth Engine helpers (init, boundaries, terrain stack, composites, downloads, zonal series)."""
import json

import ee
import pandas as pd

from . import config as C


def init():
    try:
        ee.Initialize(project=C.GEE_PROJECT or None)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.GEE_PROJECT or None)


def retry(fn, tries=6, wait=20, always=False):
    """Call fn(); on Earth Engine rate / memory / timeout errors wait and try again."""
    import time
    soft = ("Too many", "429", "timed out", "Timeout", "503", "500", "memory limit", "Connection", "Internal error")
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1 or not (always or any(s in str(e) or s in type(e).__name__ for s in soft)):
                raise
            print(f"  [retry {i + 1}/{tries - 1}] {type(e).__name__}: {str(e)[:90]}", flush=True)
            time.sleep(wait * (i + 1))


def region(bbox):
    return ee.Geometry.Rectangle(list(bbox))


def _local_districts():
    """Pak_Districts.shp (167 districts incl. Azad Kashmir and Gilgit-Baltistan) with GAUL-style column names."""
    import geopandas as gpd
    g = gpd.read_file(C.DATA / "boundaries" / "Pak_Districts.shp").to_crs(4326)
    nm = g["admin2Name"].fillna("Unnamed").str.strip()
    dup = nm.duplicated(keep=False)                          # e.g. two districts both called "Hub": make names unique
    g["ADM2_NAME"] = [f"{n} ({k})" if d else n for n, d, k in zip(nm, dup, nm.groupby(nm).cumcount() + 1)]
    g["ADM1_NAME"] = g["admin1Name"].str.title()
    g["ADM2_CODE"] = list(range(1, len(g) + 1))
    g["geometry"] = g.geometry.simplify(0.004, preserve_topology=True)     # keeps the upload small
    return g[["ADM1_NAME", "ADM2_NAME", "ADM2_CODE", "geometry"]]


def pakistan():
    import json

    from shapely.geometry import mapping
    from shapely.ops import unary_union
    u = unary_union(list(_local_districts().geometry))
    return ee.FeatureCollection([ee.Feature(ee.Geometry(mapping(u)))])


def districts(province=None):
    import json
    g = _local_districts()
    if province:
        g = g[g["ADM1_NAME"].str.lower() == province.lower()]
    return ee.FeatureCollection(json.loads(g.to_json()))


def to_gdf(fc):
    import geemap
    return geemap.ee_to_gdf(fc)


# ---------------------------------------------------------------- terrain
def terrain_stack(geom):
    """Static conditioning factors from SRTM + MERIT Hydro."""
    dem = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem)
    asp_sin = aspect.multiply(3.14159 / 180).sin().rename("aspect_sin")
    asp_cos = aspect.multiply(3.14159 / 180).cos().rename("aspect_cos")
    curv = dem.convolve(ee.Kernel.laplacian8(normalize=True)).rename("curvature")
    tri = dem.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(3)).rename("roughness")
    relief = (dem.reduceNeighborhood(ee.Reducer.max(), ee.Kernel.square(5))
              .subtract(dem.reduceNeighborhood(ee.Reducer.min(), ee.Kernel.square(5)))).rename("relief")
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa = merit.select("upa")
    hnd = merit.select("hnd").rename("hnd")
    slope_rad = slope.max(0.1).multiply(3.14159 / 180)
    twi = upa.multiply(1e6).divide(90).divide(slope_rad.tan()).log().rename("twi")
    stream = upa.gte(1)
    # MERIT Hydro is on a ~90 m grid, so pixel distance * 90 = metres
    dist_drain = stream.fastDistanceTransform(256).sqrt().multiply(90).rename("dist_drainage")
    return ee.Image.cat([dem, slope, asp_sin, asp_cos, curv, tri, relief, twi, hnd, dist_drain]).clip(geom)


def rain_mean_annual(geom, start=2000, end=2023):
    ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(f"{start}-01-01", f"{end}-12-31")
    return ic.select("precipitation").sum().divide(end - start + 1).rename("rain_annual").clip(geom)


def worldcover(geom):
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover").clip(geom)


# ---------------------------------------------------------------- optical composites
def _s2_mask(img):
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(ok).divide(10000).copyProperties(img, ["system:time_start"])


def s2_median(geom, start, end):
    ic = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(_s2_mask))
    img = ic.select(["B2", "B3", "B4", "B8", "B11", "B12"], ["blue", "green", "red", "nir", "swir1", "swir2"]).median()
    return add_indices(img).clip(geom)


def _ls_prep(img, bands_in):
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 4).eq(0))
    sr = img.select(bands_in, ["blue", "green", "red", "nir", "swir1", "swir2"]).multiply(0.0000275).add(-0.2)
    if bands_in[0] == "SR_B1":      # TM / ETM+ -> OLI equivalent (Roy et al. 2016 OLS coefficients)
        slope = ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071])
        icept = ee.Image.constant([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172])
        sr = sr.multiply(slope).add(icept).rename(["blue", "green", "red", "nir", "swir1", "swir2"])
    return sr.toFloat().updateMask(mask)        # one band type for every sensor so the collections can be merged.copyProperties(img, ["system:time_start"])


def landsat_median(geom, start, end):
    """Harmonised Landsat 5/7/8/9 Collection-2 L2 surface-reflectance median composite."""
    parts = []
    y0, y1 = int(str(start)[:4]), int(str(end)[:4])
    avail = {"LANDSAT/LT05/C02/T1_L2": (1984, 2011), "LANDSAT/LE07/C02/T1_L2": (1999, 2021),
             "LANDSAT/LC08/C02/T1_L2": (2013, 2100), "LANDSAT/LC09/C02/T1_L2": (2022, 2100)}
    for coll, bands in [("LANDSAT/LT05/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
                        ("LANDSAT/LE07/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
                        ("LANDSAT/LC08/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
                        ("LANDSAT/LC09/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])]:
        if avail[coll][1] < y0 or avail[coll][0] > y1:
            continue          # sensor not operating in this window: keeps the server graph small
        ic = ee.ImageCollection(coll).filterBounds(geom).filterDate(start, end)
        parts.append(ic.map(lambda i, b=bands: _ls_prep(i, b)))
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p)
    return add_indices(merged.median()).clip(geom)


def landsat_lst(geom, start, end, months=None):
    """Landsat 8/9 land-surface temperature (deg C) median composite + optical indices."""
    def prep(img):
        qa = img.select("QA_PIXEL")
        m = qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 4).eq(0))
        lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
        sr = (img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                         ["blue", "green", "red", "nir", "swir1", "swir2"]).multiply(0.0000275).add(-0.2))
        return sr.addBands(lst).updateMask(m).copyProperties(img, ["system:time_start"])
    ic = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
          .filterBounds(geom).filterDate(start, end))
    if months:
        ic = ic.filter(ee.Filter.calendarRange(months[0], months[1], "month"))
    return add_indices(ic.map(prep).median()).clip(geom)


def add_indices(img):
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("ndbi")
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("mndwi")
    bsi = img.expression("((s+r)-(n+b))/((s+r)+(n+b))",
                         {"s": img.select("swir1"), "r": img.select("red"),
                          "n": img.select("nir"), "b": img.select("blue")}).rename("bsi")
    return img.addBands([ndvi, ndbi, mndwi, bsi])


# ---------------------------------------------------------------- export / zonal
def download(img, tag, key, bbox, scale, overwrite=False, tile_px=None):
    """Download an ee.Image as GeoTIFF into data/rasters and write a band-name sidecar."""
    import math

    import numpy as np
    import rasterio
    import requests
    from rasterio.transform import from_origin
    C.ensure()
    path = C.DATA / "rasters" / f"{tag}_{key}.tif"
    if path.exists() and not overwrite:
        return path
    # Sequential small tiles on a fixed grid, each retried, then mosaicked: one request at a time avoids Earth
    # Engine concurrency / memory errors. The sidecar *.names.json keeps the band names.
    names = img.bandNames().getInfo()
    nb = len(names)
    NODATA = -9999.0
    img = img.toFloat().unmask(NODATA)
    dx = scale / 111319.49                                  # pixel size in degrees
    W = int(math.ceil((bbox[2] - bbox[0]) / dx))
    H = int(math.ceil((bbox[3] - bbox[1]) / dx))
    tile = tile_px or max(int(math.sqrt(28e6 / (4 * nb))), 64)   # heavy server-side maps: pass a small tile_px
    out = np.full((nb, H, W), NODATA, dtype="float32")
    ntiles = math.ceil(H / tile) * math.ceil(W / tile)
    k = 0
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            h, w = min(tile, H - r0), min(tile, W - c0)
            x0, y0 = bbox[0] + c0 * dx, bbox[3] - r0 * dx
            geom = ee.Geometry.Rectangle([x0, y0 - h * dx, x0 + w * dx, y0], proj="EPSG:4326", geodesic=False)

            def fetch():
                # region + dimensions + crsTransform pins this tile to the global grid (region alone returns 1 pixel)
                url = img.getDownloadURL({"region": geom, "dimensions": f"{w}x{h}", "crs": "EPSG:4326",
                                          "format": "GEO_TIFF", "crsTransform": [dx, 0, x0, 0, -dx, y0]})
                resp = requests.get(url, timeout=600)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                with rasterio.MemoryFile(resp.content) as mf, mf.open() as ds:
                    return ds.read()
            a = retry(fetch, tries=8, wait=30, always=True)
            hh, ww = min(a.shape[1], h), min(a.shape[2], w)
            out[:, r0:r0 + hh, c0:c0 + ww] = a[:, :hh, :ww]
            k += 1
            print(f"  [{tag}/{key}] tile {k}/{ntiles}", flush=True)
    prof = dict(driver="GTiff", height=H, width=W, count=nb, dtype="float32", crs="EPSG:4326",
                transform=from_origin(bbox[0], bbox[3], dx, dx), nodata=NODATA, compress="deflate", tiled=True,
                BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out)
    path.with_suffix(".names.json").write_text(json.dumps(names))
    return path


def zonal_timeseries(ic, fc, scale, reducer="mean", props=("ADM2_NAME",), chunk=12):
    """Per-zone, per-image statistics of an ImageCollection -> long DataFrame (date, zone props, band values)."""
    red = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()
    n = ic.size().getInfo()
    lst = ic.toList(n)
    rows = []
    for k in range(0, n, chunk):
        sub = ee.ImageCollection(lst.slice(k, k + chunk))

        def per_img(img):
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return img.reduceRegions(fc, red, scale, tileScale=4).map(lambda f: f.set("date", date))
        feats = retry(lambda: sub.map(per_img).flatten().getInfo())["features"]
        for f in feats:
            rows.append(f["properties"])
    df = pd.DataFrame(rows)
    keep = ["date", *props] + [c for c in df.columns if c not in ("date", *props)]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    return df
