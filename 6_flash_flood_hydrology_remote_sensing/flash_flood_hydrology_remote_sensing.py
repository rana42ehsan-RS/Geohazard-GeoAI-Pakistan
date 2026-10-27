"""6 - Coupling a hydrological model with remote sensing for flash-flood prediction in a data-scarce region.

Catchment : Zhob River above Zhob town (Balochistan), ungauged
Remote sensing / EO inputs : SRTM DEM (watershed, slope), ESA WorldCover + OpenLandMap soil (SCS curve number),
                             ERA5-Land soil moisture (antecedent moisture class), ERA5 hourly rainfall (Open-Meteo)
Model     : event-based SCS-CN with antecedent-moisture adjustment + SCS triangular unit hydrograph (hourly)
Calibration target : GloFAS-ERA5 modelled discharge (Open-Meteo flood API) - MODELLED, no gauge data is assumed
Run:  python flash_flood_hydrology_remote_sensing.py
"""
import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
if not hasattr(np, "in1d"):          # removed in numpy 2.4; pysheds still calls it
    np.in1d = np.isin
from pysheds.grid import Grid
from scipy.optimize import differential_evolution

from common import config as C, gee, ml, viz, webdata
from common.report import Report

TAG = "a06"
OUTLET = (69.45, 31.34)  # lon, lat near Zhob town (search centre; refined to the nearest GloFAS river cell)


def pick_reference_point():
    """GloFAS is a 0.05 deg grid: choose the neighbouring cell with the largest mean flow (= the main river)."""
    import json
    cache = C.DATA / "tables" / "a06_ref_point.json"
    if cache.exists():
        return tuple(json.loads(cache.read_text()))
    best, best_q = OUTLET, -1.0
    for dy in np.arange(-0.2, 0.201, 0.05):
        for dx in np.arange(-0.2, 0.201, 0.05):
            lon, lat = round(OUTLET[0] + dx, 3), round(OUTLET[1] + dy, 3)
            try:
                m = webdata.openmeteo_discharge(lat, lon, "2019-01-01", "2024-12-31")["q_glofas"].mean()
            except Exception:  # noqa: BLE001
                continue
            if m > best_q:
                best, best_q = (lon, lat), m
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(best))
    return best
CAL = ("2019-01-01", "2021-12-31")
VAL = ("2022-01-01", "2024-12-31")
CN_TABLE = {  # land class (WorldCover code) -> CN for HSG A, B, C, D
    10: (36, 60, 73, 79), 20: (35, 56, 70, 77), 30: (49, 69, 79, 84), 40: (67, 78, 85, 89),
    50: (89, 92, 94, 95), 60: (77, 86, 91, 94), 70: (77, 86, 91, 94), 80: (100, 100, 100, 100),
    90: (100, 100, 100, 100), 95: (100, 100, 100, 100), 100: (49, 69, 79, 84)}


def hsg(clay, sand):
    g = np.full(clay.shape, 2)                  # default C
    g[(sand > 50) & (clay < 10)] = 0            # A
    g[(sand > 40) & (clay < 20) & (g != 0)] = 1  # B
    g[clay > 40] = 3                            # D
    return g


def build_inputs(geom):
    ter = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    lc = gee.worldcover(geom).toFloat()
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("clay")
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("sand")
    return ee.Image.cat([ter, lc, clay, sand]).clip(geom)


def delineate(tif, outlet):
    arr, prof, names = ml.read_stack(tif)
    dem_path = tif.with_name(f"{TAG}_dem_only.tif")
    ml.save_raster(arr[0], prof, dem_path)
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    pit = grid.fill_pits(dem)
    fl = grid.fill_depressions(pit)
    inf = grid.resolve_flats(fl)
    fdir = grid.flowdir(inf)
    acc = grid.accumulation(fdir)
    xs, ys = grid.snap_to_mask(acc > 2000, outlet)
    catch = grid.catchment(x=xs, y=ys, fdir=fdir, xytype="coordinate")
    dist = grid.distance_to_outlet(x=xs, y=ys, fdir=fdir, xytype="coordinate")
    mask = np.asarray(catch).astype(bool)
    return arr, prof, names, mask, np.asarray(dist), np.asarray(acc), (float(np.ravel(xs)[0]), float(np.ravel(ys)[0]))


def amc_adjust(cn, amc):
    if amc == 0:
        return 4.2 * cn / (10 - 0.058 * cn)
    if amc == 2:
        return 23 * cn / (10 + 0.13 * cn)
    return cn


def runoff_series(P, amc, cn, ia_ratio, reset_h=24):
    """Hourly effective rainfall (mm) from event-based cumulative SCS-CN. amc: 0 dry,1 normal,2 wet per hour."""
    Q = np.zeros_like(P)
    cum_p = cum_q = 0.0
    dry = 0
    cn_ev = cn
    for i, p in enumerate(P):
        if p < 0.1:
            dry += 1
            if dry >= reset_h:
                cum_p = cum_q = 0.0
            continue
        if cum_p == 0.0:
            cn_ev = min(amc_adjust(cn, amc[i]), 99.5)
        dry = 0
        S = 25400 / cn_ev - 254
        ia = ia_ratio * S
        cum_p += p
        q_new = (cum_p - ia) ** 2 / (cum_p + (1 - ia_ratio) * S) if cum_p > ia else 0.0
        Q[i] = max(q_new - cum_q, 0)
        cum_q = q_new
    return Q


def uh(area_km2, tc_h, lag_factor):
    tp = max(0.6 * tc_h * lag_factor + 0.5, 1.0)
    base = int(np.ceil(2.67 * tp))
    t = np.arange(base + 1, dtype=float)
    tri = np.where(t <= tp, t / tp, np.maximum((2.67 * tp - t) / (1.67 * tp), 0))
    tri = tri / tri.sum()
    return tri * area_km2 * 1e6 * 1e-3 / 3600.0      # m3/s per mm of effective rain in one hour


def simulate(params, P, amc, cn, area, tc):
    cn_scale, ia, lagf, base = params
    qe = runoff_series(P, amc, np.clip(cn * cn_scale, 30, 98), ia)
    return np.convolve(qe, uh(area, tc, lagf))[:len(P)] + base


def nse(o, s):
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def daily(h, idx):
    return pd.Series(h, index=idx).resample("D").mean()


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    tif = gee.download(build_inputs(gee.region(cfg["bbox"])), TAG, "inputs", cfg["bbox"], cfg["scale"])
    ref_pt = pick_reference_point()
    arr, prof, names, mask, dist, acc, (ox, oy) = delineate(tif, ref_pt)
    px_m = ml.pixel_size_m(prof)
    area_km2 = mask.sum() * (px_m / 1000) ** 2
    elev = arr[0]
    L = np.nanmax(np.where(mask, dist, np.nan)) * px_m
    S = max((np.nanmax(elev[mask]) - np.nanmin(elev[mask])) / L, 1e-3)
    tc_h = 0.0195 * L ** 0.77 * S ** -0.385 / 60          # Kirpich, hours
    lc, clay, sand = arr[1], arr[2], arr[3]
    g = hsg(np.nan_to_num(clay, nan=25), np.nan_to_num(sand, nan=40))
    cn_map = np.full(lc.shape, np.nan)
    for code, row in CN_TABLE.items():
        for k in range(4):
            cn_map[(lc == code) & (g == k)] = row[k]
    cn = float(np.nanmean(cn_map[mask]))

    # ---- maps
    b = ml.bounds_of(prof)
    viz.plot_raster(np.where(mask, elev, np.nan), b, f"Zhob catchment ({area_km2:,.0f} km²)", out / "map_catchment.png",
                    cmap="terrain", label="Elevation (m)")
    viz.plot_raster(np.where(mask, cn_map, np.nan), b, "Satellite/soil-derived curve number", out / "map_cn.png",
                    cmap="YlOrBr", label="CN")

    # ---- forcing: ERA5 hourly rain + soil moisture (3 points averaged), GloFAS discharge target
    cy, cx = np.mean(np.where(mask)[0]), np.mean(np.where(mask)[1])
    clat, clon = prof["transform"].f + prof["transform"].e * cy, prof["transform"].c + prof["transform"].a * cx
    cache = C.DATA / "tables" / "a06_forcing.csv"
    if cache.exists():
        f = pd.read_csv(cache, parse_dates=["time"], index_col="time")
    else:
        dfs = [webdata.openmeteo_archive(clat + dy, clon + dx, "2019-01-01", "2024-12-31",
                                          hourly=("precipitation", "soil_moisture_0_to_7cm")).set_index("time")
               for dy, dx in [(0, 0), (0.3, 0.3), (-0.3, -0.3)]]
        f = sum(dfs) / len(dfs)
        f.to_csv(cache)
    q = webdata.openmeteo_discharge(ref_pt[1], ref_pt[0], "2019-01-01", "2024-12-31").set_index("time")["q_glofas"]
    sm = f["soil_moisture_0_to_7cm"]
    lo, hi = sm.quantile([0.33, 0.67])
    amc = np.where(sm < lo, 0, np.where(sm > hi, 2, 1))
    P = f["precipitation"].to_numpy()
    idx = f.index

    def run(period, params):
        m = (idx >= period[0]) & (idx <= period[1] + " 23:00")
        qs = simulate(params, P[m], amc[m], cn, area_km2, tc_h)
        qd = daily(qs, idx[m])
        qo = q.reindex(qd.index)
        ok = qo.notna()
        return qd[ok], qo[ok]

    def cost(p):
        qs, qo = run(CAL, p)
        return -(0.5 * nse(qo.values, qs.values) + 0.5 * nse(np.log1p(qo.values), np.log1p(qs.values)))
    de = differential_evolution(cost, bounds=[(0.55, 1.15), (0.05, 0.45), (0.3, 3.0), (0.0, float(q.quantile(0.3)) + 1)],
                                maxiter=25, popsize=10, seed=42, polish=False)
    p_best = de.x
    rows = []
    for nm, per in (("Calibration 2019-21", CAL), ("Validation 2022-24", VAL)):
        qs, qo = run(per, p_best)
        rows.append(dict(period=nm, NSE=nse(qo.values, qs.values), KGE=kge(qo.values, qs.values),
                         PBIAS_pct=100 * (qs.sum() - qo.sum()) / qo.sum(), r=np.corrcoef(qo, qs)[0, 1]))
    perf = pd.DataFrame(rows).round(3)
    perf.to_csv(out / "performance.csv", index=False)
    qs, qo = run(VAL, p_best)
    thr = q.quantile(0.95)
    ev_o, ev_s = qo > thr, qs > thr
    pod = (ev_o & ev_s).sum() / max(ev_o.sum(), 1)
    far = (~ev_o & ev_s).sum() / max(ev_s.sum(), 1)
    csi = (ev_o & ev_s).sum() / max((ev_o | ev_s).sum(), 1)
    cat = pd.DataFrame([dict(threshold_m3s=round(thr, 2), events_obs=int(ev_o.sum()), events_sim=int(ev_s.sum()),
                             POD=round(pod, 2), FAR=round(far, 2), CSI=round(csi, 2))])
    cat.to_csv(out / "flood_event_skill.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(qo.index, qo, "k", lw=1, label="GloFAS-ERA5 (modelled reference)")
    ax.plot(qs.index, qs, "tab:red", lw=1, label="SCS-CN + UH model")
    ax.axhline(thr, color="grey", ls="--", label="Q95 flash-flood threshold")
    ax.set(ylabel="m³/s", title="Validation period (daily mean)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "validation_hydrograph.png", dpi=200); plt.close(fig)

    # ---- live forecast
    fc_txt = ""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=dict(
            latitude=clat, longitude=clon, hourly="precipitation,soil_moisture_0_to_7cm", past_days=5,
            forecast_days=3, timezone="UTC"), timeout=60).json()["hourly"]
        fh = pd.DataFrame(r); fh["time"] = pd.to_datetime(fh["time"]); fh = fh.set_index("time")
        amc_f = np.where(fh["soil_moisture_0_to_7cm"] < lo, 0, np.where(fh["soil_moisture_0_to_7cm"] > hi, 2, 1))
        qf = simulate(p_best, fh["precipitation"].fillna(0).to_numpy(), amc_f, cn, area_km2, tc_h)
        peak = qf.max()
        level = "FLASH-FLOOD WARNING" if peak > thr else "Watch" if peak > 0.5 * thr else "No alert"
        fc_txt = f"Latest 3-day forecast: simulated peak {peak:.1f} m³/s vs Q95 {thr:.1f} m³/s -> {level}."
        pd.DataFrame({"time": fh.index, "rain_mm": fh["precipitation"], "q_sim": qf}).to_csv(out / "forecast_latest.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print("[warn] forecast skipped:", e)

    # ---- report
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets([f"Watershed delineation from SRTM with pysheds (outlet snapped near {ox:.3f}E, {oy:.3f}N): area {area_km2:,.0f} km², main-channel length {L/1000:.0f} km, mean slope {S:.3f} m/m, Kirpich Tc {tc_h:.1f} h.",
               f"Curve number from ESA WorldCover x hydrologic soil group (OpenLandMap sand/clay): catchment mean CN = {cn:.1f}.",
               "Antecedent moisture: ERA5-Land 0-7 cm soil moisture terciles -> AMC I / II / III adjustment of CN.",
               "Hourly rainfall: ERA5 reanalysis (Open-Meteo), average of three points in the catchment. Hourly SCS triangular unit hydrograph.",
               "Free parameters (CN multiplier, Ia/S, lag factor, baseflow) calibrated with differential evolution against GloFAS-ERA5 daily discharge, 2019-21; validated 2022-24.",
               "NOTE: the reference series is modelled, not gauged. Treat skill scores as agreement with a global model, and use any WAPDA/PMD gauge records you can obtain for a proper validation."])
    r.h("3. Results")
    r.p(f"Calibrated parameters: CN scale={p_best[0]:.2f}, Ia/S={p_best[1]:.2f}, lag factor={p_best[2]:.2f}, baseflow={p_best[3]:.2f} m³/s.")
    r.fig(out / "map_catchment.png", "Delineated catchment", 5.5)
    r.fig(out / "map_cn.png", "Curve-number map", 5.5)
    r.table(perf, "Hydrograph skill")
    r.fig(out / "validation_hydrograph.png", "Validation hydrograph", 6.3)
    r.table(cat, "Flash-flood event detection (Q95 exceedance, validation period)")
    if fc_txt:
        r.p(fc_txt)
    r.h("4. Limitations")
    r.p("Daily-aggregated comparison hides sub-daily flash-flood peaks; reanalysis rainfall underestimates convective extremes; "
        "SCS-CN is a lumped, event-based approach and ignores transmission losses in dry channels.")
    r.references(["scs", "glofas", "gee", "worldcover", "merit"])
    r.save(out / "A06_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
