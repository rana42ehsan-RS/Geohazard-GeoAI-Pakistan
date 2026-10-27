"""4 - Landslide susceptibility with hybrid ML (frequency-ratio + RF/XGBoost/stacking) and terrain analysis.

AOI: Muzaffarabad - Balakot area, the extent of the supplied landslide inventory (3,820 mapped landslides). The inventory
(shp/gpkg/geojson polygons or points, or csv with lon/lat) must be in data/inventories/. Without one the script stops:
a susceptibility model trained on proxy labels is not credible.
Run:  python landslide_susceptibility_hybrid_ml.py
"""
import ee
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import config as C, gee, ml, pipeline, viz, webdata
from common.report import Report

TAG = "a04"
# Rainfall (CHIRPS, ~5 km) is deliberately NOT a predictor: over a ~35 km AOI it only has a handful of distinct cells, so the
# model used it as a location proxy and produced 5 km block artefacts in the map and inflated hold-out scores.
FACTORS = ["elevation", "slope", "aspect_sin", "aspect_cos", "curvature", "roughness", "relief", "twi",
           "dist_drainage", "ndvi", "landcover"]


def build_stack(geom):
    ter = gee.terrain_stack(geom).select(["elevation", "slope", "aspect_sin", "aspect_cos", "curvature",
                                          "roughness", "relief", "twi", "dist_drainage"])
    rain = gee.rain_mean_annual(geom)
    ndvi = gee.s2_median(geom, "2023-04-01", "2023-10-31").select("ndvi")
    lc = gee.worldcover(geom).toFloat()
    return ee.Image.cat([ter, rain, ndvi, lc]).clip(geom)


def hybrid_models():
    xgb = lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,  # noqa: E731
                                colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=42)
    rf = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)  # noqa: E731
    return {
        "RF": rf(),
        "XGBoost": xgb(),
        "FR-RF (hybrid)": Pipeline([("fr", ml.FRTransformer()), ("m", rf())]),
        "FR-XGBoost (hybrid)": Pipeline([("fr", ml.FRTransformer()), ("m", xgb())]),
    }


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])
    tif = gee.download(build_stack(geom), TAG, "stack", cfg["bbox"], cfg["scale"], tile_px=300)
    arr, prof, names = ml.read_stack(tif)

    inv = webdata.load_landslide_inventory(cfg["bbox"])
    if inv is None or len(inv) < 30:
        raise SystemExit("This project needs a landslide inventory (>= 30 points/polygons inside the AOI) in data/inventories/.")
    ext = webdata.inventory_extent(inv)
    X, y, groups, _ = ml.dataset_from_points(arr, prof, names, inv, FACTORS, neg_ratio=1.0, extent=ext, block_m=2000)
    src = (f"mapped inventory of {len(inv)} landslides (polygons reduced to centroids; one positive sample per 30 m pixel = "
           f"{int(y.sum())}). Negatives are drawn only inside the inventory's mapped extent "
           f"({ext[0]:.2f}-{ext[2]:.2f}E, {ext[1]:.2f}-{ext[3]:.2f}N) and more than {ml.pixel_size_m(prof) * 8:.0f} m from any landslide")
    pts = (inv[:, 0], inv[:, 1])

    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets([f"Landslide labels: {src}.",
               "Inventory provenance (from its attribute table): features mapped on QuickBird / DigitalGlobe / Ikonos imagery, "
               "about 63 % observed in 2006 and 24 % in 2010, i.e. mostly earthquake-triggered (2005) and flood-season failures.",
               "Terrain analysis (SRTM 30 m): elevation, slope, aspect (sin/cos), Laplacian curvature, roughness, local relief, TWI, distance to drainage.",
               "Environmental factors: 2023 NDVI (Sentinel-2) and land cover (ESA WorldCover 2020). Rainfall is omitted on purpose: the 5 km CHIRPS grid is too coarse for a 35 km area and acted as a location proxy in a first run.",
               "Hybrid models: frequency-ratio (FR) bivariate statistics used as feature encoding for RF and XGBoost (fitted inside each training fold). Baselines: plain RF and XGBoost.",
               "Spatial block cross-validation (2 km blocks); permutation importance."])
    r.p("Caveats: the predictors describe the present landscape (2018-2023) while most inventoried failures date from 2006-2010, "
        "so vegetation and land cover may have changed since the scars were mapped; there is no ground-shaking or lithology layer; "
        "and the model is valid only inside the mapped extent, because landslides outside it were never recorded.")
    res = pipeline.run(TAG, arr, prof, names, X, y, groups, FACTORS, "Landslide", r, models=hybrid_models(), points=pts)

    # ---- terrain analysis summary: class share vs slope
    slope = arr[names.index("slope")]
    cls = res["cls"]
    import pandas as pd
    bins = [0, 10, 20, 30, 40, 90]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (slope >= lo) & (slope < hi) & np.isfinite(cls)
        rows.append(dict(Slope=f"{lo}-{hi} deg", Cells=int(m.sum()),
                         High_VeryHigh_pct=round(100 * (cls[m] >= 4).mean(), 1) if m.sum() else np.nan))
    ta = pd.DataFrame(rows)
    ta.to_csv(out / "terrain_vs_susceptibility.csv", index=False)
    viz.plot_raster(slope, ml.bounds_of(prof), "Slope (degrees)", out / "map_slope.png", cmap="magma", label="deg")
    r.h("Terrain analysis", 2)
    r.fig(out / "map_slope.png", "Slope map of the study area", 5.5)
    r.table(ta, "Share of high/very-high susceptibility by slope class")

    r.h("3. Discussion and limitations")
    r.p("Hybrid FR-ML models usually add interpretability by encoding how strongly each factor class is associated with failures; "
        "compare the FR hybrids against the plain learners in the metrics table before concluding that they perform better. "
        "Lithology, fault proximity and earthquake shaking are missing here because no open, consistent layers were used - add them when available.")
    r.references(["gee", "fr", "rf", "xgb", "chirps", "worldcover"])
    r.save(out / "A04_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
