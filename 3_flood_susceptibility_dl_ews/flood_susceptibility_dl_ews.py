"""3 - Deep learning + GIS flood susceptibility and early warning (Lower Indus, Sindh).

Labels : Sentinel-1 flood extent of Aug-Sep 2022 (change detection) minus permanent water
Factors: terrain, hydrology, rainfall, soil, land cover, NDVI
Models : RF / XGBoost / stacking baseline  vs  patch-based CNN (PyTorch)
Early warning: 7-day Open-Meteo rainfall forecast x susceptibility -> alert level per grid cell
Run:  python flood_susceptibility_dl_ews.py
"""
import ee
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import config as C, gee, ml, pipeline, viz, webdata
from common.report import Report

TAG = "a03"
# Rainfall (CHIRPS, ~5 km) is NOT a predictor: it varies in 5 km steps and acts as a location code, leaving square-edged patches
# in the susceptibility map and inflating hold-out scores.
FACTORS = ["elevation", "slope", "curvature", "twi", "hnd", "dist_drainage", "relief", "clay", "ndvi", "landcover"]
PATCH = 11
torch.manual_seed(42)


def build_stack(geom):
    ter = gee.terrain_stack(geom).select(["elevation", "slope", "curvature", "twi", "hnd", "dist_drainage", "relief"])
    rain = gee.rain_mean_annual(geom)
    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate("2000-01-01", "2021-12-31").select("precipitation")
    max3 = daily.reduce(ee.Reducer.max()).rename("rain_max_daily")  # extreme daily rain 2000-2021 (pre-event)
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("clay")
    ndvi = gee.s2_median(geom, "2021-01-01", "2021-12-31").select("ndvi")
    lc = gee.worldcover(geom)
    # --- flood label from Sentinel-1 (2022) - pre-flood baseline
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV")
    # light speckle reduction via a 3x3 mean after compositing (much cheaper than focal_median on every tile)
    pre = s1.filterDate("2022-03-01", "2022-06-15").median().focal_mean(1, "square", "pixels")
    post = s1.filterDate("2022-08-25", "2022-09-20").median().focal_mean(1, "square", "pixels")
    diff = post.subtract(pre)
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").unmask(0).gte(10)
    flood = post.lt(-16).And(diff.lt(-3)).And(perm.Not()).And(ter.select("slope").lt(5))
    nofl = post.gte(-16).And(diff.gt(-1)).And(perm.Not())
    label = ee.Image(0).where(nofl, 0).where(flood, 1).updateMask(flood.Or(nofl)).rename("flood")
    return ee.Image.cat([ter, rain, max3, clay, ndvi, lc.toFloat(), label]).clip(geom)


# ------------------------------------------------------------------ CNN
class PatchCNN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def patches(norm, rows, cols, k=PATCH):
    h = k // 2
    pad = np.pad(norm, ((0, 0), (h, h), (h, h)), mode="reflect")
    return np.stack([pad[:, r:r + k, c:c + k] for r, c in zip(rows, cols)]).astype("float32")


def train_cnn(norm, rows, cols, y, groups, epochs=25):
    tr, te = ml.spatial_split(groups)
    Xtr, Xte = patches(norm, rows[tr], cols[tr]), patches(norm, rows[te], cols[te])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCNN(norm.shape[0]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(Xtr), torch.tensor(y[tr], dtype=torch.float32)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(xt[b].to(dev)), yt[b].to(dev))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(Xte).to(dev))).cpu().numpy()
        hist.append(dict(epoch=ep + 1, loss=float(loss), auc_test=roc_auc_score(y[te], p)))
    return model, pd.DataFrame(hist), p, te


def predict_cnn(model, norm, valid, step=4096):
    dev = next(model.parameters()).device
    H, W = valid.shape
    out = np.full((H, W), np.nan, dtype="float32")
    rr, cc = np.where(valid)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rr), step):
            r, c = rr[i:i + step], cc[i:i + step]
            p = torch.sigmoid(model(torch.tensor(patches(norm, r, c)).to(dev))).cpu().numpy()
            out[r, c] = p
    return out


def early_warning(prob_cls, prof, out):
    """Combine static susceptibility class with the 7-day rainfall forecast on a coarse grid."""
    b = ml.bounds_of(prof)
    lons = np.linspace(b[0] + 0.05, b[2] - 0.05, 8)
    lats = np.linspace(b[1] + 0.05, b[3] - 0.05, 8)
    LL, LA = np.meshgrid(lons, lats)
    fc = webdata.openmeteo_forecast(LA.ravel(), LL.ravel(), 7)
    r3 = fc.groupby(["lat", "lon"])["precipitation_sum"].apply(lambda s: s.rolling(3).sum().max()).reset_index(name="rain3d_max")
    t = prof["transform"]
    cols = ((r3["lon"] - t.c) / t.a).astype(int).clip(0, prob_cls.shape[1] - 1)
    rows = ((r3["lat"] - t.f) / t.e).astype(int).clip(0, prob_cls.shape[0] - 1)
    r3["susc_class"] = prob_cls[rows, cols]
    # alert matrix: rain thresholds (mm / 3 days) tightened for higher susceptibility
    def level(r):
        k = r["susc_class"] if np.isfinite(r["susc_class"]) else 1
        thr = [100 - 10 * (k - 1), 60 - 7 * (k - 1), 30 - 4 * (k - 1)]  # mm / 3 days: severe, warning, watch
        return 3 if r.rain3d_max >= thr[0] else 2 if r.rain3d_max >= thr[1] else 1 if r.rain3d_max >= thr[2] else 0
    r3["alert"] = r3.apply(level, axis=1)
    r3["alert_text"] = r3["alert"].map({0: "None", 1: "Watch", 2: "Warning", 3: "Severe"})
    r3.to_csv(out / "early_warning_latest.csv", index=False)
    return r3


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])
    tif = gee.download(build_stack(geom), TAG, "stack", cfg["bbox"], cfg["scale"])
    arr, prof, names = ml.read_stack(tif)

    X, y, groups, (rows, cols) = ml.dataset_from_label_band(arr, prof, names, "flood", FACTORS, n_per_class=8000)
    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Flood labels: Sentinel-1 VV backscatter, post-flood median (25 Aug - 20 Sep 2022) < -16 dB and >3 dB drop vs the Mar-Jun 2022 median, excluding permanent water (JRC) and slopes >5 degrees; non-flood = post-flood backscatter >= -16 dB and change > -1 dB. Radar-derived labels have their own errors - see limitations.",
               "Ten conditioning factors (terrain, HAND, TWI, distance to drainage, clay, NDVI, land cover). Rainfall is left out on purpose: the 5 km CHIRPS grid acts as a location code.",
               "Baselines: RF, XGBoost, extra-trees and a stacking ensemble. Deep model: 11x11-pixel patch CNN (3 conv blocks) trained with AdamW.",
               "Spatial block hold-out for all models so that CNN and baselines are compared on unseen locations."])
    res = pipeline.run(TAG, arr, prof, names, X, y, groups, FACTORS, "Flood", r)

    # CNN on standardised stack
    F = np.stack([arr[names.index(f)] for f in FACTORS])
    mu, sd = np.nanmean(F, axis=(1, 2), keepdims=True), np.nanstd(F, axis=(1, 2), keepdims=True) + 1e-6
    norm = np.nan_to_num((F - mu) / sd, nan=0.0).astype("float32")
    cnn, hist, p_te, te = train_cnn(norm, rows, cols, y, groups)
    hist.to_csv(out / "cnn_training.csv", index=False)
    valid = np.isfinite(F).all(axis=0)
    cnn_prob = predict_cnn(cnn, norm, valid)
    ml.save_raster(cnn_prob, prof, out / "flood_susceptibility_cnn.tif")
    cls, _ = ml.classify_quantiles(cnn_prob)
    b = ml.bounds_of(prof)
    viz.plot_raster(cnn_prob, b, "Flood susceptibility - patch CNN", out / "map_cnn.png", cmap="RdYlBu_r", vmin=0, vmax=1,
                    label="Probability")
    cnn_auc = hist["auc_test"].iloc[-1]
    rf_auc = res["metrics"].set_index("model").loc[res["best"], "AUC_test"]
    r.h("CNN versus classical ML", 2)
    r.p(f"The patch CNN reached a hold-out AUC of {cnn_auc:.3f} versus {rf_auc:.3f} for the best classical model "
        f"({res['best']}). Spatial context in 11x11 windows {'improved' if cnn_auc > rf_auc else 'did not improve'} discrimination in this experiment.")
    r.table(hist.tail(8).round(3), "CNN learning curve (last epochs)")
    r.fig(out / "map_cnn.png", "CNN flood susceptibility", 5.8)

    # early warning
    try:
        ew = early_warning(cls, prof, out)
        r.h("3. Early warning prototype", 1)
        r.p("Each cell of an 8x8 grid receives the 7-day Open-Meteo rainfall forecast; alert thresholds on 3-day rainfall "
            "are lowered for higher susceptibility classes. This is a prototype decision rule that must be tuned on observed flood events before operational use.")
        r.table(ew.sort_values("alert", ascending=False).head(10).round(2), "Highest alerts in latest forecast")
    except Exception as e:  # noqa: BLE001
        print("[warn] early-warning step skipped:", e)

    r.h("4. Limitations")
    r.p("Sentinel-1 flood labels can include false positives (smooth surfaces, wet agriculture) and miss floods under dense vegetation/urban areas. "
        "Rainfall predictors are climatological, so the model maps where floods tend to occur, not the event timing.")
    r.references(["gee", "sentinel1", "gsw", "merit", "cnn", "rf", "xgb"])
    r.save(out / "A03_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
