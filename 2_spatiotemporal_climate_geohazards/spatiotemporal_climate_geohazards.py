"""2 - Spatiotemporal modeling of climate-induced geohazards (Indus basin), ML + Earth observation.

(1) Mann-Kendall / Sen trends of monsoon rain, temperature, snow cover (2000-2024)
(2) Gridded monsoon-rain trend map (CHIRPS, pixel-wise linear fit)
(3) ML model of monthly floodplain surface-water fraction (JRC) from lagged climate drivers
Run:  python spatiotemporal_climate_geohazards.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from common import config as C, gee, ml, viz
from common.report import Report

TAG = "a02"
PLAIN = [66.0, 24.0, 72.5, 33.0]      # Punjab/Sindh floodplain (response)
UPPER = [70.5, 31.5, 77.5, 37.0]      # upper Indus catchment (drivers)
Y0, Y1 = 2000, 2024
NATIONAL_BBOX = [60.8, 23.6, 77.9, 37.1]     # Pakistan, extent of the rainfall-trend map


def monthly_mean(ic, band, geom, scale, name):
    def f(img):
        v = img.select(band).reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e10, tileScale=4).get(band)
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-01"), name: v})
    n = ic.size().getInfo()
    lst = ic.toList(n)
    rows = []
    for k in range(0, n, 24):
        rows += [x["properties"] for x in gee.retry(lambda: ee.ImageCollection(lst.slice(k, k + 24)).map(f).getInfo())["features"]]
    return pd.DataFrame(rows)


def water_fraction(geom):
    ic = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").filterDate(f"{Y0}-01-01", f"{Y1}-12-31")

    def f(img):
        obs = img.gt(0)
        wat = img.eq(2)
        n_o = obs.reduceRegion(ee.Reducer.sum(), geom, 300, maxPixels=1e10, tileScale=8).get("water")
        n_w = wat.reduceRegion(ee.Reducer.sum(), geom, 300, maxPixels=1e10, tileScale=8).get("water")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-01"), "obs": n_o, "wat": n_w})
    rows = []
    n = ic.size().getInfo()
    lst = ic.toList(n)
    for k in range(0, n, 4):
        rows += [x["properties"] for x in gee.retry(lambda: ee.ImageCollection(lst.slice(k, k + 4)).map(f).getInfo())["features"]]
    d = pd.DataFrame(rows)
    d["wfrac"] = d["wat"] / d["obs"].replace(0, np.nan)
    d.loc[d["obs"] < 0.4 * d["obs"].max(), "wfrac"] = np.nan     # discard poorly observed months
    return d[["date", "wfrac"]]


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    up, pl = gee.region(UPPER), gee.region(PLAIN)
    cache = C.DATA / "tables" / "a02_monthly.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
    else:
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(f"{Y0}-01-01", f"{Y1}-12-31"))
        months = ee.List.sequence(0, (Y1 - Y0 + 1) * 12 - 1)
        pm = ee.ImageCollection(months.map(lambda m: chirps.filterDate(
            ee.Date.fromYMD(Y0, 1, 1).advance(m, "month"), ee.Date.fromYMD(Y0, 1, 1).advance(ee.Number(m).add(1), "month"))
            .sum().rename("precip").set("system:time_start", ee.Date.fromYMD(Y0, 1, 1).advance(m, "month").millis())))
        era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(f"{Y0}-01-01", f"{Y1}-12-31")
        p_up = monthly_mean(pm, "precip", up, 5000, "p_upper")
        p_pl = monthly_mean(pm, "precip", pl, 5000, "p_plain")
        t_up = monthly_mean(era, "temperature_2m", up, 11000, "t_upper")
        s_up = monthly_mean(era, "snow_cover", up, 11000, "snow_upper")
        w = water_fraction(pl)
        df = w
        for d in (p_up, p_pl, t_up, s_up):
            df = df.merge(d, on="date", how="outer")
        df["date"] = pd.to_datetime(df["date"])
        df["t_upper"] -= 273.15
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache, index=False)

    df = df.set_index("date")
    # ---------------- trends
    rows = []
    for col, label in [("p_upper", "Monsoon rain upper basin (JJAS)"), ("p_plain", "Monsoon rain floodplain (JJAS)"),
                       ("t_upper", "Temperature upper basin (JJAS)"), ("snow_upper", "Snow cover upper basin (MAM)")]:
        months = [6, 7, 8, 9] if "JJAS" in label else [3, 4, 5]
        s = df[df.index.month.isin(months)][col].groupby(df[df.index.month.isin(months)].index.year)
        s = (s.sum() if col.startswith("p_") else s.mean()).dropna()
        res = mk.original_test(s.values)
        rows.append(dict(Variable=label, Trend=res.trend, p_value=round(res.p, 4), Sen_slope_per_year=round(res.slope, 3),
                         tau=round(res.Tau, 3)))
    trends = pd.DataFrame(rows)
    trends.to_csv(out / "mann_kendall.csv", index=False)

    # ---------------- gridded monsoon-rain trend map
    yrs = ee.List.sequence(Y0, Y1)
    chirps_d = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    def yr(y):
        s = chirps_d.filterDate(ee.Date.fromYMD(y, 6, 1), ee.Date.fromYMD(y, 9, 30)).sum()
        return s.rename("p").addBands(ee.Image.constant(y).float().rename("t")).set("year", y)
    fit = ee.ImageCollection(yrs.map(yr)).select(["t", "p"]).reduce(ee.Reducer.linearFit()).select("scale").rename("trend_mm_yr")
    tif = gee.download(fit.clip(gee.pakistan().geometry()), TAG, "monsoon_trend", NATIONAL_BBOX, 5000)
    arr, prof, names = ml.read_stack(tif)
    viz.plot_raster(arr[0], ml.bounds_of(prof), "Trend of JJAS rainfall 2000-2024 (CHIRPS)", out / "map_rain_trend.png",
                    cmap="BrBG", vmin=-6, vmax=6, label="mm per year")

    # ---------------- ML for floodplain water fraction
    d = df.copy()
    d["month"] = d.index.month
    d["msin"], d["mcos"] = np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)
    feats = ["msin", "mcos"]
    for c in ["p_upper", "p_plain", "t_upper", "snow_upper"]:
        for lag in range(0, 4):
            d[f"{c}_l{lag}"] = d[c].shift(lag)
            feats.append(f"{c}_l{lag}")
    d["p_upper_3m"] = d["p_upper"].rolling(3).sum()
    feats.append("p_upper_3m")
    d = d.dropna(subset=feats + ["wfrac"])
    X, y = d[feats], d["wfrac"]
    models = {"RandomForest": RandomForestRegressor(400, min_samples_leaf=2, random_state=42, n_jobs=-1),
              "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42)}
    res, preds = [], {}
    tscv = TimeSeriesSplit(5)
    for n, m in models.items():
        oof = pd.Series(np.nan, index=y.index)
        for a, b in tscv.split(X):
            m.fit(X.iloc[a], y.iloc[a])
            oof.iloc[b] = m.predict(X.iloc[b])
        ok = oof.notna()
        res.append(dict(model=n, R2=r2_score(y[ok], oof[ok]), RMSE=mean_squared_error(y[ok], oof[ok]) ** 0.5))
        preds[n] = oof
    res = pd.DataFrame(res).round(3)
    best = res.sort_values("R2", ascending=False).iloc[0]["model"]
    res.to_csv(out / "ml_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y, label="Observed (JRC)", color="k", lw=1)
    ax.plot(preds[best].index, preds[best], label=f"{best} (walk-forward)", color="tab:red", lw=1)
    ax.set(ylabel="Surface-water fraction", title="Floodplain surface-water fraction: observed vs predicted")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "ts_obs_pred.png", dpi=200); plt.close(fig)

    m = models[best].fit(X, y)
    imp = pd.Series(m.feature_importances_, index=feats).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax, color="#2b6c8f")
    ax.set(title="Driver importance", xlabel="Importance")
    fig.tight_layout(); fig.savefig(out / "driver_importance.png", dpi=200); plt.close(fig)

    ann = df.groupby(df.index.year)["wfrac"].max()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(ann.index, ann.values, color="#2b6c8f")
    for yv in (2010, 2022):
        ax.annotate(str(yv), (yv, ann.get(yv, 0)), ha="center", va="bottom", color="red")
    ax.set(title="Annual maximum floodplain water fraction (JRC)", ylabel="Fraction")
    fig.tight_layout(); fig.savefig(out / "annual_max_water.png", dpi=200); plt.close(fig)

    # ---------------- report
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["CHIRPS (precipitation), ERA5-Land (temperature, snow cover), JRC Global Surface Water monthly history (flood response), all via Google Earth Engine.",
               "Mann-Kendall test and Sen slope for monotonic trends; pixel-wise linear fit for the rainfall trend map.",
               "Random forest and XGBoost on lagged (0-3 months) drivers and seasonality; evaluated with walk-forward time-series cross-validation (no leakage from the future)."])
    r.h("3. Results")
    r.table(trends, "Mann-Kendall trend results (2000-2024)")
    r.fig(out / "map_rain_trend.png", "Spatial trend of monsoon rainfall", 5.5)
    r.table(res, "Walk-forward model skill")
    r.fig(out / "ts_obs_pred.png", "Observed and predicted floodplain water fraction", 6.3)
    r.fig(out / "driver_importance.png", "Most important climate drivers", 4.8)
    r.fig(out / "annual_max_water.png", "Annual maxima (2010 and 2022 flood years annotated)", 6)
    r.h("4. Limitations")
    r.p("JRC monthly water is cloud-limited during the monsoon (months with poor observation are masked), so extreme peaks "
        "are under-sampled. Drivers are basin means; the model captures hydro-climatic association, not physical routing.")
    r.references(["gee", "chirps", "gsw", "mk", "sen", "rf", "xgb"])
    r.save(out / "A02_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
