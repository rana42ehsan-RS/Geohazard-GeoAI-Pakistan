"""5 - Impact of LULC change on natural-hazard dynamics: multi-decadal analysis (Swat valley).

Landsat 5/7/8/9 composites for 2000, 2005, 2010, 2015, 2020. A SEPARATE random forest is trained for every epoch with
labels that exist for that year (Hansen forest cover/loss, GHSL built-up, JRC yearly water, MODIS land cover), which avoids
transferring one year's spectral signatures to another sensor/year. -> change matrices -> Markov projection to 2040
-> SCS-CN runoff response -> flood-extent vs rainfall regression.
Run:  python lulc_change_hazard_dynamics.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import config as C, gee, ml, viz
from common.report import Report

TAG = "a05"
EPOCHS = [2000, 2005, 2010, 2015, 2020]
CLASSES = {1: "Tree cover", 2: "Shrub/grass", 3: "Cropland", 4: "Built-up", 5: "Bare/snow", 6: "Water"}
COLORS = ["#1b7837", "#a6d96a", "#fee08b", "#d73027", "#bdbdbd", "#2b83ba"]
CN = {1: 70, 2: 74, 3: 82, 4: 94, 5: 91, 6: 100}      # SCS curve numbers, hydrologic soil group C (assumption)
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndbi", "mndwi", "bsi", "elevation", "slope"]


def labels_for(y):
    """Reference classes for year y from independent, year-specific products (0/unlabelled pixels are masked).

    Priority water > built-up > tree > MODIS-derived classes. Reference labels are coarse (100-500 m), so the
    per-epoch accuracy reported is agreement with these products, not an independent field validation.
    """
    h = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    # Hansen masks the zero (no loss / no gain) pixels, so unmask them or every logical test drops those pixels
    tc, loss, gain = (h.select("treecover2000").unmask(0), h.select("lossyear").unmask(0), h.select("gain").unmask(0))
    alive = loss.eq(0).Or(loss.gt(y - 2000))
    tree = tc.gte(30).And(alive)
    if y >= 2012:
        tree = tree.Or(gain.eq(1).And(alive))
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{y}").select("built_surface").gte(1500)
    water = (ee.ImageCollection("JRC/GSW1_4/YearlyHistory").filter(ee.Filter.eq("year", min(y, 2021)))
             .first().select("waterClass").eq(3))
    my = max(2001, min(y, 2022))
    modis = (ee.ImageCollection("MODIS/061/MCD12Q1").filterDate(f"{my}-01-01", f"{my}-12-31").first()
             .select("LC_Type1"))
    grass = modis.gte(6).And(modis.lte(10))
    crop = modis.eq(12).Or(modis.eq(14))
    bare = modis.eq(15).Or(modis.eq(16))
    cls = ee.Image(0).where(grass, 2).where(crop, 3).where(bare, 5).where(tree, 1).where(built, 4).where(water, 6)
    return cls.selfMask().rename("class")


def composite(geom, year):
    """Harmonised Landsat composite + terrain (terrain stabilises snow/bare/built-up separation across sensors)."""
    dem = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    return (gee.landsat_median(geom, f"{year-1}-03-01", f"{year+1}-10-31").select(BANDS[:-2])
            .addBands(dem).addBands(ee.Terrain.slope(dem).rename("slope")).clip(geom))


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])
    maps, acc_rows = {}, []
    for y in EPOCHS:
        comp = composite(geom, y)
        samples = comp.addBands(labels_for(y).clip(geom)).stratifiedSample(
            numPoints=800, classBand="class", region=geom, scale=90, seed=42, geometries=False,
            tileScale=8).randomColumn("r")
        train, test = samples.filter(ee.Filter.lt("r", 0.7)), samples.filter(ee.Filter.gte("r", 0.7))
        clf = ee.Classifier.smileRandomForest(150, seed=42, minLeafPopulation=3).train(train, "class", BANDS)
        cm = test.classify(clf, "pred").errorMatrix("class", "pred")
        acc_rows.append(dict(year=y, overall_accuracy=gee.retry(lambda: cm.accuracy().getInfo()),
                             kappa=gee.retry(lambda: cm.kappa().getInfo())))
        img = comp.classify(clf).rename("class").focal_mode(1, "square", "pixels").toFloat()   # 3x3 majority filter
        tif = gee.download(img, TAG, f"lulc_{y}", cfg["bbox"], cfg["scale"], tile_px=250)
        maps[y] = ml.read_stack(tif)[0][0]
    acc = pd.DataFrame(acc_rows).round(3)
    acc.to_csv(out / "classification_agreement.csv", index=False)
    _, prof, _ = ml.read_stack(C.DATA / "rasters" / f"{TAG}_lulc_{EPOCHS[0]}.tif")
    px_km2 = (ml.pixel_size_m(prof) / 1000) ** 2
    b = ml.bounds_of(prof)

    # ---- area table + maps
    area = pd.DataFrame({y: [(maps[y] == k).sum() * px_km2 for k in CLASSES] for y in EPOCHS}, index=CLASSES.values()).round(1)
    area.to_csv(out / "lulc_area_km2.csv")

    # ---- cross-check classified areas against the reference products themselves (tree: Hansen, built-up: GHSL)
    def ref_area(y, k):
        m = labels_for(y).clip(geom).eq(k).rename("a").multiply(ee.Image.pixelArea())
        v = gee.retry(lambda: m.reduceRegion(ee.Reducer.sum(), geom, 90, maxPixels=1e10, tileScale=8).get("a").getInfo())
        return round((v or 0) / 1e6, 1)
    chk = pd.DataFrame({"year": EPOCHS,
                        "Tree_reference_km2": [ref_area(y, 1) for y in EPOCHS],
                        "Tree_classified_km2": [area.loc["Tree cover", y] for y in EPOCHS],
                        "Built_reference_km2": [ref_area(y, 4) for y in EPOCHS],
                        "Built_classified_km2": [area.loc["Built-up", y] for y in EPOCHS]})
    chk["Tree_bias_pct"] = (100 * (chk.Tree_classified_km2 / chk.Tree_reference_km2 - 1)).round(1)
    chk["Built_bias_pct"] = (100 * (chk.Built_classified_km2 / chk.Built_reference_km2 - 1)).round(1)
    chk.to_csv(out / "reference_vs_classified_km2.csv", index=False)

    for y in EPOCHS:
        viz.plot_classes(maps[y], b, f"LULC {y}", out / f"map_lulc_{y}.png", labels=list(CLASSES.values()), colors=COLORS)
    fig, ax = plt.subplots(figsize=(8, 4))
    area.T.plot.bar(stacked=True, ax=ax, color=COLORS)
    ax.set(ylabel="km²", title="LULC composition through time"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(out / "area_stack.png", dpi=200); plt.close(fig)

    # ---- transition matrix first -> last epoch and Markov projection
    a, z = maps[EPOCHS[0]].ravel(), maps[EPOCHS[-1]].ravel()
    ok = np.isfinite(a) & np.isfinite(z)
    T = pd.crosstab(pd.Series(a[ok].astype(int)).map(CLASSES), pd.Series(z[ok].astype(int)).map(CLASSES)) * px_km2
    T.round(1).to_csv(out / f"transition_{EPOCHS[0]}_{EPOCHS[-1]}_km2.csv")
    P = T.div(T.sum(axis=1), axis=0).reindex(index=CLASSES.values(), columns=CLASSES.values()).fillna(0)
    from scipy.linalg import fractional_matrix_power
    s0 = area[EPOCHS[-1]].to_numpy()
    # P describes one (first->last epoch) step; the step from the last epoch to 2040 is its fractional power
    Pf = np.real(fractional_matrix_power(P.to_numpy() + 1e-9 * np.eye(len(P)), (2040 - EPOCHS[-1]) / (EPOCHS[-1] - EPOCHS[0])))
    proj2040 = pd.Series(s0 @ Pf, index=CLASSES.values()).clip(lower=0).round(1)
    proj2040.to_csv(out / "markov_projection_2040_km2.csv")

    # ---- SCS-CN runoff response (design storm 100 mm)
    rows = []
    for y in EPOCHS:
        m = maps[y]
        cn = np.full(m.shape, np.nan)
        for k, v in CN.items():
            cn[m == k] = v
        S = 25400 / cn - 254
        P_ = 100.0
        Q = np.where(P_ > 0.2 * S, (P_ - 0.2 * S) ** 2 / (P_ + 0.8 * S), 0)
        rows.append(dict(year=y, mean_CN=np.nanmean(cn), runoff_mm=np.nanmean(Q), runoff_coeff=np.nanmean(Q) / P_))
    cn_df = pd.DataFrame(rows).round(3)
    cn_df.to_csv(out / "scs_cn_runoff.csv", index=False)

    # ---- flood extent vs rainfall and LULC (annual, 2000-2021; JRC monthly history ends 2021)
    gsw = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").filter(ee.Filter.calendarRange(6, 9, "month"))
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    valley = geom

    def per_year(yv):
        yv = ee.Number(yv)
        yr = gsw.filter(ee.Filter.calendarRange(yv, yv, "year"))
        zero = ee.Image.constant(0).rename("water")          # years without scenes give a defined empty result
        w = ee.Image(ee.Algorithms.If(yr.size().gt(0), yr.map(lambda i: i.eq(2)).max().rename("water"), zero))
        obs = ee.Image(ee.Algorithms.If(yr.size().gt(0), yr.map(lambda i: i.gt(0)).max().rename("water"), zero))
        aw = w.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), valley, 90, maxPixels=1e10).get("water")
        ao = obs.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), valley, 90, maxPixels=1e10).get("water")
        p = rain.filterDate(ee.Date.fromYMD(yv, 6, 1), ee.Date.fromYMD(yv, 9, 30)).sum() \
            .reduceRegion(ee.Reducer.mean(), valley, 5000).get("precipitation")
        return ee.Feature(None, {"year": yv, "water_m2": aw, "obs_m2": ao, "rain": p})
    ts = pd.DataFrame([gee.retry(lambda y=y: per_year(y).getInfo(), tries=6, wait=15)["properties"]
                       for y in range(2000, 2022)])
    ts["water_frac"] = ts.water_m2 / ts.obs_m2.replace(0, np.nan)
    built = np.interp(ts.year, EPOCHS, chk.Built_reference_km2.values / area.sum().values * 100)   # GHSL reference: the classified built-up class is biased high
    ts["built_pct"] = built
    ts.to_csv(out / "flood_extent_annual.csv", index=False)
    d = ts.dropna()
    if len(d) > 8:
        Xr = sm.add_constant(d[["rain", "built_pct"]])
        ols = sm.OLS(d["water_frac"], Xr).fit()
        reg = pd.DataFrame({"coef": ols.params, "p_value": ols.pvalues}).round(5)
        r2 = ols.rsquared
    else:
        reg, r2 = pd.DataFrame(), np.nan

    # ---- Hansen forest loss
    h = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss = h.select("lossyear").gt(0).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), geom, 60, maxPixels=1e10).get("lossyear")
    loss_km2 = ee.Number(loss).divide(1e6).getInfo()

    # ---- report
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Landsat Collection-2 surface reflectance (TM, ETM+, OLI) median composites, +/-1 year around each epoch.",
               "One random forest (150 trees) per epoch, trained on that year's Landsat composite (+ SRTM elevation/slope) with year-specific reference labels: Hansen tree cover/loss/gain (tree), GHSL built-up surface (built-up), JRC yearly water (water), MODIS MCD12Q1 (grass/shrub, cropland, bare/snow). Landsat 5/7 reflectance is transformed to OLI equivalents (Roy et al. 2016).",
               "Post-classification change detection, transition matrix, Markov chain projection to 2040.",
               "SCS curve-number runoff for a 100 mm design storm (hydrologic soil group C assumed everywhere).",
               "OLS of annual flood-season (JJAS) surface-water fraction on JJAS rainfall and built-up share; Hansen GFC forest loss."])
    r.h("3. Results")
    r.p("Agreement with the reference products on a 30 % hold-out per epoch (not an independent field validation, because the reference labels are themselves coarse):")
    r.table(acc, "Per-epoch classification agreement")
    r.p("Reference-product areas (Hansen tree cover net of loss; GHSL built-up surface) are compared with the classified areas below. "
        "A large bias means the classifier, not the landscape, drives that class's trend, and the reference series should be preferred for that class.")
    r.table(chk, "Classified versus reference areas (km²)")
    r.p("Built-up is over-estimated by the classifier (low-density settlements are confused with bare land and cropland), so the GHSL series is used as the built-up trend in the flood regression and should be quoted for urban growth; the classified built-up map is only indicative of where settlements are.")
    for y in EPOCHS:
        r.fig(out / f"map_lulc_{y}.png", f"LULC {y}", 4.3)
    r.fig(out / "area_stack.png", "Class area through time", 5.5)
    r.table(area.reset_index().rename(columns={"index": "Class"}), "Area (km²) per class and epoch")
    r.table(T.round(1).reset_index().rename(columns={"index": "From/To"}), f"Transition matrix {EPOCHS[0]} to {EPOCHS[-1]} (km²)")
    r.table(proj2040.reset_index().rename(columns={"index": "Class", 0: "km2_2040"}), "Markov projection for 2040 (km²)")
    r.table(cn_df, "Runoff response for a 100 mm storm")
    if len(reg):
        r.table(reg.reset_index().rename(columns={"index": "term"}), f"OLS on flood-season water fraction (R²={r2:.2f})")
    r.p(f"Hansen GFC indicates {loss_km2:.0f} km² of tree-cover loss within the AOI during 2001-2023 (includes plantation cycles).")
    r.h("4. Limitations")
    r.p("Reference labels are coarse (100-500 m) and the classes are partly defined by them, so agreement scores overstate true accuracy; "
        "report change only where it exceeds map error. Flood regression is correlative (annual n=22, cloud-limited water observations).")
    r.references(["gee", "worldcover", "rf", "scs", "markov", "hansen", "gsw", "chirps"])
    r.save(out / "A05_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
