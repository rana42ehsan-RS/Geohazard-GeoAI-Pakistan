"""1 - AI-Driven Multi-Hazard Risk Assessment (remote sensing + geospatial big data), Pakistan, district level.

Hazards (flood, drought, heat, landslide, earthquake) x Exposure (people, built-up, lights) x Vulnerability proxy.
AI part: PCA, K-means typology (k chosen by silhouette), Isolation Forest outliers, entropy-weighted index.
Run:  python multihazard_risk.py
"""
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from common import config as C, gee, viz, webdata
from common.report import Report

TAG = "a01"


def entropy_weights(M):
    P = M / (M.sum(axis=0) + 1e-12)
    E = -(P * np.log(P + 1e-12)).sum(axis=0) / np.log(len(M))
    d = 1 - E
    return d / d.sum()


def fill_missing(g, cols, zero_cols=("pop",)):
    """Districts with no valid satellite pixels (small / glaciated) return null: fill with the median, pop with 0.
    Returns the number of districts filled per column so the report can state it."""
    n = g[cols].isna().sum()
    for c in cols:
        g[c] = g[c].fillna(0 if c in zero_cols else g[c].median())
    return n[n > 0].to_dict()


def pct(s):
    return pd.Series(rankdata(s) / len(s), index=s.index)


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    dist = gee.districts()
    geom = gee.pakistan().geometry()

    # ---------------- indicators (all national, 1 km)
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood = merit.select("hnd").lt(10).And(slope.lt(2)).rename("flood_prone")           # low-lying flat land
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0).rename("water_ever")
    years = ee.List.sequence(1991, 2023)
    ann = ee.ImageCollection(years.map(lambda y: ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                                       .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
                                       .sum().set("year", y)))
    rain_cv = ann.reduce(ee.Reducer.stdDev()).divide(ann.mean().max(1)).rename("rain_cv")  # drought proneness
    aridity = ann.mean().rename("rain_mean")
    heat = (ee.ImageCollection("MODIS/061/MOD11A2").filterDate("2015-01-01", "2024-12-31")
            .filter(ee.Filter.calendarRange(6, 8, "month")).select("LST_Day_1km").mean()
            .multiply(0.02).subtract(273.15).rename("lst_summer"))
    steep = slope.gt(20).rename("steep_share")
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("year", 2020)).mosaic().rename("pop")
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").divide(1e4).rename("built_share")
    lights = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filterDate("2022-01-01", "2023-12-31")
              .select("avg_rad").mean().rename("lights"))
    means = ee.Image.cat([flood, gsw, rain_cv, aridity, heat, steep, built, lights]).clip(geom)
    fc = means.reduceRegions(dist, ee.Reducer.mean(), 1000, tileScale=8)
    fc_pop = pop.reduceRegions(dist, ee.Reducer.sum(), 500, tileScale=8)
    g = gee.to_gdf(fc)
    gp = gee.to_gdf(fc_pop)[["ADM2_CODE", "sum"]].rename(columns={"sum": "pop"})
    g = g.merge(gp, on="ADM2_CODE")
    IND = ["flood_prone", "water_ever", "rain_cv", "rain_mean", "lst_summer", "steep_share", "built_share", "lights", "pop"]
    filled = fill_missing(g, IND)
    if filled:
        print("[note] districts filled with the median (no valid pixels):", filled)
    g["area_km2"] = g.to_crs(32642).area / 1e6
    g["pop_density"] = g["pop"] / g["area_km2"]

    # ---------------- seismic hazard proxy from the USGS catalogue
    cat = webdata.usgs_catalog(cfg["bbox"], "1973-01-01", "2025-12-31", 4.5)
    cen = g.to_crs(32642).geometry.centroid.to_crs(4326)
    lat, lon = np.radians(cen.y.values)[:, None], np.radians(cen.x.values)[:, None]
    la2, lo2 = np.radians(cat.latitude.values)[None, :], np.radians(cat.longitude.values)[None, :]
    d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - lat) / 2) ** 2 + np.cos(lat) * np.cos(la2) * np.sin((lo2 - lon) / 2) ** 2))
    energy = 10 ** (1.5 * cat.mag.values)[None, :]
    g["seismic_energy"] = np.log10((energy * np.exp(-d / 75)).sum(axis=1) + 1)  # distance-decayed energy

    # ---------------- build indices
    H = pd.DataFrame({
        "flood": pct(g["flood_prone"] + g["water_ever"]),
        "drought": pct(g["rain_cv"] + 1 / (g["rain_mean"] + 10)),
        "heat": pct(g["lst_summer"]),
        "landslide": pct(g["steep_share"]),
        "earthquake": pct(g["seismic_energy"]),
    })
    E = pd.DataFrame({"pop": pct(g["pop"]), "density": pct(g["pop_density"]), "built": pct(g["built_share"])})
    V = pd.DataFrame({"low_lights": pct(-g["lights"]), "low_built": pct(-g["built_share"])})
    wH, wE, wV = (entropy_weights(x.to_numpy() + 0.01) for x in (H, E, V))
    g["hazard"] = H.to_numpy() @ wH
    g["exposure"] = E.to_numpy() @ wE
    g["vulnerability"] = V.to_numpy() @ wV
    g["risk"] = pct(g["hazard"] * g["exposure"] * g["vulnerability"])
    for c in H:
        g[f"H_{c}"] = H[c]
    g["multi_hazard_count"] = (H > 0.8).sum(axis=1)

    # ---------------- AI: PCA + KMeans typology + Isolation Forest
    Z = StandardScaler().fit_transform(pd.concat([H, E, V], axis=1))
    pcs = PCA(n_components=3).fit(Z)
    g[["pc1", "pc2", "pc3"]] = pcs.transform(Z)
    best_k, best_s = 4, -1
    from sklearn.metrics import silhouette_score
    for k in range(3, 8):
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        s = silhouette_score(Z, km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    g["typology"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(Z).astype(str)
    g["outlier"] = IsolationForest(random_state=42, contamination=0.05).fit_predict(Z) == -1

    g.drop(columns=["geometry"]).to_csv(out / "district_risk_table.csv", index=False)
    g.to_file(out / "district_risk.gpkg", driver="GPKG")
    g.to_file(out / "district_risk.geojson", driver="GeoJSON")

    # ---------------- figures
    for col, name in [("hazard", "Composite hazard"), ("exposure", "Exposure"), ("vulnerability", "Vulnerability"),
                      ("risk", "Multi-hazard risk")]:
        viz.choropleth(g, col, f"{name} index by district", out / f"map_{col}.png", label=col)
    for c in H:
        viz.choropleth(g, f"H_{c}", f"{c.capitalize()} hazard index", out / f"map_H_{c}.png", label=c)
    viz.choropleth(g, "multi_hazard_count", "Number of hazards in top quintile", out / "map_count.png",
                   scheme="equal_interval", k=6, cmap="Reds")
    viz.choropleth(g, "typology", f"AI-derived risk typologies (k={best_k})", out / "map_typology.png",
                   cmap="Set2", scheme="unique_values")
    top = g.sort_values("risk", ascending=False).head(15)[["ADM1_NAME", "ADM2_NAME", "hazard", "exposure",
                                                           "vulnerability", "risk", "multi_hazard_count"]].round(3)
    top.to_csv(out / "top15_districts.csv", index=False)

    # ---------------- report
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Flood: share of low-HAND (<10 m) flat land (MERIT Hydro, SRTM) + maximum surface-water extent (JRC GSW).",
               "Drought: coefficient of variation and aridity of annual rainfall, 1991-2023 (CHIRPS).",
               "Heat: mean summer daytime land surface temperature 2015-2024 (MODIS MOD11A2).",
               "Landslide: share of terrain steeper than 20 degrees (SRTM).",
               "Earthquake: distance-decayed seismic energy from USGS M>=4.5 events 1973-2025.",
               "Exposure: population (WorldPop 2020), density, built-up share (GHSL). Vulnerability proxy: low night-light and low built-up (low capacity/ development proxy).",
               "Indicators are percentile-ranked and combined with entropy weights; Risk = Hazard x Exposure x Vulnerability.",
               "AI: PCA, K-means (k chosen by silhouette), Isolation Forest outlier detection."])
    r.h("3. Results")
    r.p(f"Entropy weights (hazard): " + ", ".join(f"{c}={w:.2f}" for c, w in zip(H.columns, wH)) +
        f". PCA explained {pcs.explained_variance_ratio_[:3].sum()*100:.1f}% of variance in three components. "
        f"K-means found {best_k} typologies (silhouette {best_s:.2f}).")
    for f, c in [("map_hazard", "Composite hazard"), ("map_exposure", "Exposure"), ("map_vulnerability", "Vulnerability"),
                 ("map_risk", "Multi-hazard risk"), ("map_count", "Count of high-level hazards"),
                 ("map_typology", "Risk typologies")]:
        r.fig(out / f"{f}.png", f"{c} by district", 5.5)
    r.table(top, "Fifteen highest-risk districts")
    if filled:
        r.p("Data gaps: " + ", ".join(f"{k}: {v} districts" for k, v in filled.items()) + " had no valid satellite pixels and were filled with the national median (population with 0).")
    r.h("4. Limitations")
    r.p("Indicators are proxies, not loss data; vulnerability lacks socio-economic census variables; "
        "weights are data-driven (entropy), not expert-elicited. Validate against NDMA/PDMA loss records.")
    r.references(["gee", "chirps", "gsw", "merit", "worldpop", "xgb"])
    r.save(out / "A01_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
