"""9 - Satellite-based drought monitoring and prediction with multi-index fusion and AI (Balochistan, district level).

Indices : SPI-3/6 (CHIRPS), VCI (MODIS NDVI), TCI (MODIS LST), soil-moisture z-score and PDSI (TerraClimate)
Fusion  : first principal component of the standardised indices = Combined Drought Index (CDI)
AI      : RF / XGBoost forecast of CDI 1 and 3 months ahead; time-based train/test split; persistence baseline
Run:  python drought_monitoring_prediction.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from common import config as C, gee, viz
from common.report import Report

TAG = "a09"
Y0, Y1 = 1991, 2023
IDX = ["spi3", "spi6", "vci", "tci", "sm_z", "pdsi"]


def monthly_collection(fc_geom):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    ndvi = ee.ImageCollection("MODIS/061/MOD13A3").select("NDVI")
    lst = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km")
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(["soil", "pdsi"])
    months = ee.List.sequence(0, (Y1 - Y0 + 1) * 12 - 1)

    empty = ee.Image.constant(0).updateMask(ee.Image.constant(0))      # fully masked -> null statistic

    def mean_or_empty(col, s, e, band, mul, add, name):
        c = col.filterDate(s, e).select(band)
        return ee.Image(ee.Algorithms.If(c.size().gt(0), c.mean().multiply(mul).add(add).rename(name),
                                         empty.rename(name)))

    def f(m):
        s = ee.Date.fromYMD(Y0, 1, 1).advance(m, "month")
        e = s.advance(1, "month")
        p = chirps.filterDate(s, e).sum().rename("precip")
        n = mean_or_empty(ndvi, s, e, "NDVI", 0.0001, 0, "ndvi")
        l = mean_or_empty(lst, s, e, "LST_Day_1km", 0.02, -273.15, "lst")
        so = mean_or_empty(tc, s, e, "soil", 1, 0, "soil")
        pd_ = mean_or_empty(tc, s, e, "pdsi", 1, 0, "pdsi")
        return ee.Image.cat([p, n, l, so, pd_]).set("system:time_start", s.millis())
    return ee.ImageCollection(months.map(f))


def spi(series, k):
    """SPI over k months per calendar month using gamma fit (zeros handled by mixed distribution)."""
    roll = series.rolling(k).sum()
    out = pd.Series(np.nan, index=series.index)
    for m in range(1, 13):
        v = roll[roll.index.month == m].dropna()
        pos = v[v > 0]
        if len(pos) < 10:
            continue
        a, loc, scale = stats.gamma.fit(pos, floc=0)
        q = (len(v) - len(pos)) / len(v)
        cdf = q + (1 - q) * stats.gamma.cdf(v.clip(lower=1e-6), a, loc=0, scale=scale)
        out[v.index] = stats.norm.ppf(np.clip(cdf, 1e-4, 1 - 1e-4))
    return out


def minmax_by_month(s, invert=False):
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        v = s[s.index.month == m]
        lo, hi = v.min(), v.max()
        z = (v - lo) / (hi - lo + 1e-9)
        out[v.index] = (1 - z) if invert else z
    return out * 100


def zscore_by_month(s):
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        v = s[s.index.month == m]
        out[v.index] = (v - v.mean()) / (v.std() + 1e-9)
    return out


def indices(d):
    d = d.sort_values("date").set_index("date")
    o = pd.DataFrame(index=d.index)
    o["spi3"], o["spi6"] = spi(d["precip"], 3), spi(d["precip"], 6)
    o["vci"] = minmax_by_month(d["ndvi"].where(d.index >= "2000-03-01"))
    o["tci"] = minmax_by_month(d["lst"].where(d.index >= "2000-03-01"), invert=True)
    o["sm_z"] = zscore_by_month(d["soil"])
    o["pdsi"] = d["pdsi"] / 100.0 if d["pdsi"].abs().max() > 50 else d["pdsi"]
    return o


def drought_class(z):
    return pd.cut(z, [-np.inf, -2, -1.5, -1, np.inf], labels=["Extreme", "Severe", "Moderate", "None/Mild"])


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    dist = gee.districts("Balochistan")
    cache = C.DATA / "tables" / "a09_district_monthly.csv"
    if cache.exists():
        raw = pd.read_csv(cache, parse_dates=["date"])
    else:
        raw = gee.zonal_timeseries(monthly_collection(None), dist, 5000, props=("ADM2_NAME",))
        raw.to_csv(cache, index=False)

    frames = []
    for name, g in raw.groupby("ADM2_NAME"):
        ix = indices(g)
        ix["district"] = name
        frames.append(ix.reset_index())
    D = pd.concat(frames, ignore_index=True).dropna(subset=["vci", "tci", "spi3", "spi6", "sm_z"])
    D["pdsi"] = D["pdsi"].fillna(0)

    # ---- fusion by PCA (aligned so that high = wet / healthy)
    Z = (D[IDX] - D[IDX].mean()) / D[IDX].std()
    pca = PCA(n_components=1).fit(Z)
    load = pca.components_[0] * (1 if pca.components_[0].mean() > 0 else -1)
    D["cdi"] = (Z.to_numpy() @ load) * (1 if pca.components_[0].mean() > 0 else -1)
    D["cdi"] = (D["cdi"] - D["cdi"].mean()) / D["cdi"].std()
    D["class"] = drought_class(D["cdi"])
    loadings = pd.DataFrame({"index": IDX, "pc1_loading": load.round(3)})
    loadings.to_csv(out / "pca_loadings.csv", index=False)
    D.to_csv(out / "district_monthly_indices.csv", index=False)

    # ---- figures: time series + heat map
    prov = D.groupby("date")["cdi"].mean()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(prov.index, prov, 0, where=prov < 0, color="#d7191c", alpha=0.7)
    ax.fill_between(prov.index, prov, 0, where=prov >= 0, color="#2c7bb6", alpha=0.7)
    ax.axhline(-1, color="k", ls=":", lw=0.8)
    ax.set(title="Balochistan mean Combined Drought Index (CDI, z-score)", ylabel="CDI")
    fig.tight_layout(); fig.savefig(out / "cdi_timeseries.png", dpi=200); plt.close(fig)
    hm = D.pivot_table(index="district", columns="date", values="cdi")
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap="RdBu", vmin=-2.5, vmax=2.5,
                   extent=(mdates(hm.columns[0]), mdates(hm.columns[-1]), len(hm), 0))
    ax.set_yticks(np.arange(len(hm)) + 0.5); ax.set_yticklabels(hm.index, fontsize=6)
    ax.xaxis_date(); fig.colorbar(im, label="CDI"); ax.set_title("CDI by district and month")
    fig.tight_layout(); fig.savefig(out / "cdi_heatmap.png", dpi=200); plt.close(fig)

    # ---- AI forecasting (1 and 3 months ahead) with time-based split
    res, best_models = [], {}
    D = D.sort_values(["district", "date"]).reset_index(drop=True)
    for h in (1, 3):
        d = D.copy()
        d["target"] = d.groupby("district")["cdi"].shift(-h)
        for c in IDX + ["cdi"]:
            for lag in (1, 2, 3):
                d[f"{c}_l{lag}"] = d.groupby("district")[c].shift(lag)
        d["msin"], d["mcos"] = np.sin(2 * np.pi * d.date.dt.month / 12), np.cos(2 * np.pi * d.date.dt.month / 12)
        feats = [c for c in d.columns if c not in ("date", "district", "class", "target")]
        d = d.dropna(subset=feats + ["target"])
        tr, te = d[d.date <= "2016-12-31"], d[d.date >= "2018-01-01"]
        base = te["cdi"]
        for nm, mdl in (("RandomForest", RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=42)),
                        ("XGBoost", XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=42))):
            mdl.fit(tr[feats], tr["target"])
            p = mdl.predict(te[feats])
            ev, evp = te["target"] < -1, p < -1
            res.append(dict(horizon=f"{h} month", model=nm, R2=r2_score(te["target"], p),
                            RMSE=mean_squared_error(te["target"], p) ** 0.5,
                            POD=(ev & evp).sum() / max(ev.sum(), 1), FAR=(~ev & evp).sum() / max(evp.sum(), 1)))
            best_models[(h, nm)] = (mdl, feats)
        evp = base < -1
        ev = te["target"] < -1
        res.append(dict(horizon=f"{h} month", model="Persistence", R2=r2_score(te["target"], base),
                        RMSE=mean_squared_error(te["target"], base) ** 0.5,
                        POD=(ev & evp).sum() / max(ev.sum(), 1), FAR=(~ev & evp).sum() / max(evp.sum(), 1)))
    perf = pd.DataFrame(res).round(3)
    perf.to_csv(out / "forecast_skill.csv", index=False)

    # ---- latest-state map and 1-month forecast
    last = D[D.date == D.date.max()].set_index("district")
    gdf = gee.to_gdf(dist).rename(columns={"ADM2_NAME": "district"}).merge(last[["cdi"]], on="district", how="left")
    viz.choropleth(gdf, "cdi", f"Combined Drought Index, {D.date.max():%b %Y}", out / "map_cdi_latest.png",
                   cmap="RdBu", scheme="quantiles", label="CDI")
    gdf.to_file(out / "district_cdi_latest.geojson", driver="GeoJSON")
    worst = D[D["class"].isin(["Severe", "Extreme"])].groupby(D.date.dt.year).size()
    worst.to_csv(out / "severe_extreme_district_months_by_year.csv")

    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Precipitation CHIRPS (SPI-3, SPI-6 by gamma fit per calendar month, 1991-2023 baseline).",
               "Vegetation: MODIS MOD13A3 NDVI -> Vegetation Condition Index (VCI). Temperature: MODIS MOD11A2 8-day 1 km daytime LST, averaged per month -> Temperature Condition Index (TCI).",
               "Soil moisture and PDSI from TerraClimate (soil-moisture z-score per calendar month).",
               "Fusion: PCA on six standardised indices; PC1 (sign-aligned so that low = dry) is the Combined Drought Index (CDI).",
               "Drought classes: CDI < -1 moderate, < -1.5 severe, < -2 extreme.",
               "Prediction: random forest and XGBoost on current and lagged (1-3 months) indices plus seasonality for 1- and 3-month horizons; trained on 1991/2001-2016, tested on 2018-2023; persistence is the benchmark."])
    r.h("3. Results")
    r.p(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of the variance in the six indices. VCI/TCI exist only after Mar 2000, so the fused index covers 2000-2023.")
    r.table(loadings, "PCA loadings of the fused index")
    r.fig(out / "cdi_timeseries.png", "Province mean CDI", 6.3)
    r.fig(out / "cdi_heatmap.png", "CDI by district and month", 6.3)
    r.fig(out / "map_cdi_latest.png", "Latest CDI map", 5.3)
    r.table(perf, "Forecast skill on the 2018-2023 test period")
    r.h("4. Limitations")
    r.p("District means mix irrigated and rangeland pixels; MODIS LST is cloud-limited; TerraClimate soil moisture is a modelled product. "
        "Forecast skill that does not beat persistence indicates the model adds little beyond autocorrelation - report this honestly.")
    r.references(["spi", "vci", "terraclimate", "chirps", "gee", "rf", "xgb"])
    r.save(out / "A09_report.docx")
    print("Done ->", out)


def mdates(ts):
    import matplotlib.dates as md
    return md.date2num(pd.Timestamp(ts).to_pydatetime())


if __name__ == "__main__":
    main()
