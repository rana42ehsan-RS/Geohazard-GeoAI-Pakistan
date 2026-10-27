"""10 - Urban heat islands as emerging climate hazards (Lahore) with remote sensing and spatial analytics.

Landsat 8/9 LST (summer 2014-2025), SUHI intensity, Getis-Ord Gi* hot spots, trend, RF driver model,
heat-hazard x population exposure x vegetation-deficit vulnerability.
Run:  python urban_heat_island_lahore.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import ndimage
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from common import config as C, gee, ml, viz
from common.report import Report

TAG = "a10"
YEARS = list(range(2014, 2026))
DRIVERS = ["ndvi", "ndbi", "mndwi", "bsi", "elevation", "built_frac", "pop_density"]


def gistar_lattice(x, valid):
    """Getis-Ord Gi* z-scores on a regular grid: binary 8-neighbour weights that include the focal cell.

    z = (sum_w x - xbar * Wi) / (S * sqrt((n * Wi - Wi^2) / (n - 1))); with 0/1 weights sum(w^2) = Wi. Cells without data
    (valid == False) carry no weight. Implemented directly with a 3x3 convolution so it can be checked by hand.
    """
    from scipy.ndimage import convolve
    k = np.ones((3, 3))
    v = valid.astype(float)
    xv = np.where(valid, x, 0.0)
    n = v.sum()
    xbar = xv.sum() / n
    S = np.sqrt((xv ** 2).sum() / n - xbar ** 2)
    Wi = convolve(v, k, mode="constant")
    sx = convolve(xv, k, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (sx - xbar * Wi) / (S * np.sqrt((n * Wi - Wi ** 2) / (n - 1)))
    return np.where(valid, z, np.nan)


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])

    # ---- summer composite (May-Sep, 2019-2025 pooled) with drivers
    comp = gee.landsat_lst(geom, "2019-01-01", "2025-12-31", months=(5, 9))
    comp = comp.select(["lst", "ndvi", "ndbi", "mndwi", "bsi"])
    dem = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").divide(1e4).rename("built_frac")      # fraction (0-1) of each 100 m cell that is built
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("year", 2020)).mosaic().divide(0.01).rename("pop_density")
    lc = gee.worldcover(geom).toFloat()
    stack = ee.Image.cat([comp, dem, built, pop, lc]).clip(geom)
    tif = gee.download(stack, TAG, "summer_stack", cfg["bbox"], cfg["scale"])
    arr, prof, names = ml.read_stack(tif)
    lst = arr[names.index("lst")]
    lc_a = arr[names.index("landcover")]
    b = ml.bounds_of(prof)

    # ---- SUHI: urban (built-up) minus rural reference (grass/cropland more than ~900 m (30 px) from any built-up pixel)
    urban = lc_a == 50
    rural = np.isin(lc_a, [30, 40]) & ~ndimage.binary_dilation(urban, iterations=30)
    suhi = np.nanmean(lst[urban]) - np.nanmean(lst[rural])
    pix = ml.pixel_size_m(prof)
    viz.plot_raster(lst, b, "Summer land surface temperature, Lahore (May-Sep 2019-25)", out / "map_lst.png",
                    cmap="inferno", label="°C")
    anom = lst - np.nanmean(lst[rural])
    viz.plot_raster(anom, b, "LST anomaly vs rural reference", out / "map_lst_anomaly.png", cmap="coolwarm",
                    vmin=-8, vmax=8, label="°C")

    # ---- LST by land cover
    lc_names = {10: "Trees", 30: "Grass", 40: "Cropland", 50: "Built-up", 60: "Bare", 80: "Water"}
    tab = pd.DataFrame([dict(Class=n, Mean_LST=np.nanmean(lst[lc_a == k]), SD=np.nanstd(lst[lc_a == k]),
                             Pixels=int((lc_a == k).sum())) for k, n in lc_names.items() if (lc_a == k).sum() > 0]).round(2)
    tab.to_csv(out / "lst_by_landcover.csv", index=False)

    # ---- hot spots (Gi*) on a 300 m aggregation
    k = max(int(300 / pix), 1)
    Hh, Ww = (lst.shape[0] // k) * k, (lst.shape[1] // k) * k
    agg = np.nanmean(lst[:Hh, :Ww].reshape(Hh // k, k, Ww // k, k), axis=(1, 3))
    ok = np.isfinite(agg)
    z = gistar_lattice(agg, ok)
    sig = np.where(~ok, np.nan, np.where(z > 2.58, 3, np.where(z > 1.96, 2, np.where(z > 1.65, 1, np.where(z < -1.96, -1, 0))))).astype(float)
    cls_lab = ["Cold spot (95%)", "Not sig.", "Hot 90%", "Hot 95%", "Hot 99%"]
    cm = np.where(np.isfinite(sig), sig + 2, np.nan)
    viz.plot_classes(cm, b, "Getis-Ord Gi* LST hot spots (300 m)", out / "map_hotspots.png", labels=cls_lab,
                     colors=["#2c7bb6", "#eeeeee", "#fdae61", "#f46d43", "#a50026"])
    hot_pct = 100 * np.nansum(sig >= 2) / np.isfinite(sig).sum()

    # ---- trend 2014-2025 (annual summer mean LST, urban vs rural) via GEE
    def yr(y):
        c = gee.landsat_lst(geom, ee.Date.fromYMD(y, 5, 1), ee.Date.fromYMD(y, 9, 30), None).select("lst")
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(geom)
        u = c.updateMask(wc.eq(50)).reduceRegion(ee.Reducer.mean(), geom, 90, maxPixels=1e10).get("lst")
        r = c.updateMask(wc.eq(40)).reduceRegion(ee.Reducer.mean(), geom, 90, maxPixels=1e10).get("lst")
        return ee.Feature(None, {"year": y, "urban": u, "rural": r})
    ts = pd.DataFrame([f["properties"] for f in ee.FeatureCollection([yr(y) for y in YEARS]).getInfo()["features"]]).dropna()
    ts["suhi"] = ts.urban - ts.rural
    ts.to_csv(out / "suhi_trend.csv", index=False)
    tr = mk.original_test(ts.suhi.values) if len(ts) > 5 else None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts.year, ts.urban, "o-", color="tab:red", label="Built-up")
    ax.plot(ts.year, ts.rural, "o-", color="tab:green", label="Cropland")
    ax.plot(ts.year, ts.suhi, "s--", color="k", label="SUHI (urban - rural)")
    ax.set(ylabel="°C", title="Summer (May-Sep) mean LST"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "trend.png", dpi=200); plt.close(fig)

    # ---- RF driver model (spatial block split)
    F = np.stack([arr[names.index(f)] for f in DRIVERS])
    valid = np.isfinite(F).all(axis=0) & np.isfinite(lst)
    rr, cc = np.where(valid)
    rng = np.random.default_rng(42)
    sel = rng.choice(len(rr), min(40000, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    X = pd.DataFrame(F[:, rr, cc].T, columns=DRIVERS)
    y = lst[rr, cc]
    grp = (rr // max(int(2000 / pix), 1)) * 10000 + cc // max(int(2000 / pix), 1)
    tri, tei = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(X, y, grp))
    rf = RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X.iloc[tri], y[tri])
    r2 = r2_score(y[tei], rf.predict(X.iloc[tei]))
    pi = permutation_importance(rf, X.iloc[tei][:5000], y[tei][:5000], n_repeats=5, random_state=42)
    imp = pd.DataFrame({"driver": DRIVERS, "importance": pi.importances_mean}).sort_values("importance", ascending=False).round(4)
    imp.to_csv(out / "driver_importance.csv", index=False)

    # ---- heat risk = hazard x exposure x vulnerability (percentile, pixel level)
    def pr(a):
        v = a[np.isfinite(a)]
        return np.where(np.isfinite(a), np.searchsorted(np.sort(v), a) / len(v), np.nan)
    hazard = pr(lst)
    expo = pr(arr[names.index("pop_density")])
    vuln = pr(-arr[names.index("ndvi")])
    risk = hazard * expo * vuln
    rc, _ = ml.classify_quantiles(risk)
    ml.save_raster(risk, prof, out / "heat_risk.tif")
    ml.save_raster(lst, prof, out / "lst_summer.tif")
    viz.plot_classes(rc, b, "Urban heat risk (hazard x exposure x vulnerability)", out / "map_heat_risk.png")

    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Landsat 8/9 Collection-2 Level-2 ST_B10 land surface temperature, cloud-masked median, May-Sep 2019-2025 (30 m; native 100 m thermal resampled).",
               "Indices: NDVI, NDBI, MNDWI, BSI. Built-up fraction (GHSL 2020), WorldPop 2020 density, SRTM elevation, ESA WorldCover.",
               "SUHI = mean LST of built-up minus mean LST of cropland farther than ~1 km from built-up land.",
               "Getis-Ord Gi* on 300 m cells (8-neighbour binary weights incl. the focal cell, closed-form z-scores); Mann-Kendall trend of annual SUHI; RF model of LST with spatial-block validation.",
               "Heat risk = LST percentile x population-density percentile x vegetation-deficit percentile."])
    r.h("3. Results")
    r.p(f"Mean summer SUHI = {suhi:.2f} °C. {hot_pct:.1f}% of cells are significant hot spots (Gi* z > 1.96, 95 % confidence). "
        f"RF explains R² = {r2:.2f} of LST on unseen spatial blocks." + (f" SUHI trend: {tr.trend}, Sen slope {tr.slope:.3f} °C/yr (p={tr.p:.3f})." if tr else ""))
    r.fig(out / "map_lst.png", "Summer LST", 5.6); r.fig(out / "map_lst_anomaly.png", "LST anomaly", 5.6)
    r.fig(out / "map_hotspots.png", "Gi* hot spots", 5.6)
    r.table(tab, "LST by land cover"); r.fig(out / "trend.png", "Annual urban/rural LST and SUHI", 5)
    r.table(imp, "RF driver importance"); r.fig(out / "map_heat_risk.png", "Heat risk classes", 5.6)
    r.h("4. Limitations")
    r.p("Satellite LST is skin temperature at ~10:30-11:00 local time, not air temperature or night-time heat stress. Cloud-free summer "
        "scenes are limited by monsoon cover and dust. Vulnerability uses only vegetation deficit; add age, income and health data when available.")
    r.references(["oke", "gee", "worldpop", "getis", "mk", "rf", "worldcover"])
    r.save(out / "A10_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
