"""11 - GeoAI framework for forest-fire susceptibility mapping and risk mitigation (Margalla - Murree - Galiyat).

Labels : MODIS MCD64A1 burned area 2001-2023 (500 m) inside vegetated land (WorldCover tree/shrub/grass)
Factors: terrain, pre-season NDVI / moisture, LST, rainfall, distance to built-up (human ignition), land cover
Models : RF / XGBoost / ExtraTrees / stacking with spatial block validation
Mitigation: K-means placement of watch-tower / patrol hubs on very-high-risk cells weighted by wildland-urban proximity
Run:  python forest_fire_geoai.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import boundaries, config as C, gee, ml, pipeline, viz
from common.report import Report

TAG = "a11"
# The 5 km pre-season rainfall layer is NOT a predictor: it acts as a location code (square-edged patches, inflated scores).
FACTORS = ["elevation", "slope", "aspect_sin", "aspect_cos", "twi", "ndvi", "ndbi", "lst", "dist_built", "landcover"]


def build_stack(geom):
    ter = gee.terrain_stack(geom).select(["elevation", "slope", "aspect_sin", "aspect_cos", "twi"])
    pre = gee.landsat_lst(geom, "2019-03-01", "2024-05-31", months=(3, 5)).select(["ndvi", "ndbi", "lst"])
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filter(ee.Filter.calendarRange(3, 5, "month"))
            .filterDate("2001-01-01", "2023-12-31").select("precipitation").sum().divide(23).rename("rain_preseason"))
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").gt(500)
    dist_built = built.selfMask().fastDistanceTransform(256).sqrt().multiply(100).rename("dist_built")
    lc = gee.worldcover(geom).toFloat()
    veg = lc.eq(10).Or(lc.eq(20)).Or(lc.eq(30))
    burned = (ee.ImageCollection("MODIS/061/MCD64A1").filterDate("2001-01-01", "2023-12-31").select("BurnDate")
              .map(lambda i: i.gt(0)).max())
    label = ee.Image(0).where(burned.eq(1), 1).updateMask(veg).rename("burned")
    return ee.Image.cat([ter, pre, rain, dist_built, lc, label]).clip(geom)


def burn_stats(geom):
    ic = ee.ImageCollection("MODIS/061/MCD64A1").filterDate("2001-01-01", "2023-12-31").select("BurnDate")

    def f(img):
        a = img.gt(0).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), geom, 500, maxPixels=1e9).get("BurnDate")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "area_m2": a})
    rows = [x["properties"] for x in ic.map(f).getInfo()["features"]]
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    d["km2"] = d["area_m2"].fillna(0) / 1e6
    return d


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])
    tif = gee.download(build_stack(geom), TAG, "stack", cfg["bbox"], cfg["scale"])
    arr, prof, names = ml.read_stack(tif)
    X, y, groups, _ = ml.dataset_from_label_band(arr, prof, names, "burned", FACTORS, n_per_class=6000)

    bs = burn_stats(geom)
    seas = bs.groupby(bs.date.dt.month)["km2"].sum()
    ann = bs.groupby(bs.date.dt.year)["km2"].sum()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].bar(seas.index, seas.values, color="#d95f0e"); axs[0].set(title="Burned area by month (2001-23)", xlabel="Month", ylabel="km²")
    axs[1].bar(ann.index, ann.values, color="#8c2d04"); axs[1].set(title="Burned area by year", xlabel="Year")
    fig.tight_layout(); fig.savefig(out / "burn_seasonality.png", dpi=200); plt.close(fig)

    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Fire labels: MODIS MCD64A1 burned-area product (500 m), any burn 2001-2023, restricted to vegetated land (ESA WorldCover tree/shrub/grass). Non-burned vegetated pixels are the negative class. The 500 m product misses small fires typical of this region, so negatives contain some undetected burns.",
               "Predictors: SRTM terrain, TWI, pre-fire-season (Mar-May) Landsat NDVI, NDBI and LST, distance to built-up land (human ignition proxy), land cover. Pre-season rainfall (CHIRPS, 5 km) is deliberately excluded because it acts as a location code over this small area.",
               "Spatial block validation; RF, XGBoost, ExtraTrees and a stacking ensemble; permutation importance.",
               "Mitigation: K-means (k = 8) on the coordinates of very-high-susceptibility cells weighted by proximity to settlements -> candidate watch-tower / patrol hubs."])
    res = pipeline.run(TAG, arr, prof, names, X, y, groups, FACTORS, "Forest fire", r)

    # ---- mitigation
    cls, prob = res["cls"], res["prob"]
    d_b = arr[names.index("dist_built")]
    prio = np.where(cls >= 5, prob * np.exp(-d_b / 3000.0), np.nan)
    rr, cc = np.where(np.isfinite(prio) & (prio > np.nanquantile(prio, 0.5)))
    wts = prio[rr, cc]
    km = KMeans(8, n_init=10, random_state=42).fit(np.c_[rr, cc], sample_weight=wts)
    t = prof["transform"]
    hubs = pd.DataFrame({"hub": range(1, 9), "lon": t.c + t.a * (km.cluster_centers_[:, 1] + 0.5),
                         "lat": t.f + t.e * (km.cluster_centers_[:, 0] + 0.5)})
    hubs["cells_served"] = np.bincount(km.labels_, minlength=8)
    hubs.round(4).to_csv(out / "mitigation_hubs.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    b = ml.bounds_of(prof)
    ax.imshow(cls, extent=(b[0], b[2], b[1], b[3]), cmap="YlOrRd", vmin=1, vmax=5)
    ax.scatter(hubs.lon, hubs.lat, marker="^", s=130, c="navy", edgecolor="w", label="Proposed watch/patrol hubs")
    for _, h in hubs.iterrows():
        ax.annotate(int(h.hub), (h.lon, h.lat), xytext=(4, 4), textcoords="offset points", color="navy", fontsize=9)
    ax.set(title="Susceptibility classes and proposed mitigation hubs", xlabel="Longitude", ylabel="Latitude")
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    boundaries.add_context(ax, b)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out / "map_mitigation.png", dpi=220); plt.close(fig)

    r.h("3. Seasonality and mitigation", 1)
    r.fig(out / "burn_seasonality.png", "Burned area by month and year", 6.2)
    peak = int(seas.idxmax())
    r.p(f"Burned area peaks in month {peak}; patrols, controlled burning of fuel load and public awareness should be concentrated before this window.")
    r.fig(out / "map_mitigation.png", "Candidate hub locations", 5.6)
    r.table(hubs.round(4), "Proposed hub coordinates (to be checked against road access and existing forest-department posts)")
    r.bullets(["Fuel management and fire lines along very-high-susceptibility ridges adjacent to settlements.",
               "Early-warning by combining this map with daily fire-weather forecasts .",
               "Community patrols and awareness in the wildland-urban interface (Murree, Galiyat, Margalla)."])
    r.h("4. Limitations")
    r.p("MODIS burned area detects fires of roughly 25 ha or more; small surface fires in chir-pine litter are under-represented. "
        "Human ignition and fuel-type (pine vs broadleaf) are only proxied. Hub placement is a decision-support suggestion, not an engineering design.")
    r.references(["giglio", "gee", "worldcover", "chirps", "rf", "xgb"])
    r.save(out / "A11_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
