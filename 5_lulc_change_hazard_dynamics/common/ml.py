"""Raster-based susceptibility machine learning with spatial cross-validation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

SEED = 42


def read_stack(tif):
    """-> (array[B,H,W] float32 with NaN nodata, rasterio profile, band names)."""
    tif = Path(tif)
    with rasterio.open(tif) as ds:
        arr = ds.read().astype("float32")
        prof = ds.profile
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == ds.nodata] = np.nan
    side = tif.with_suffix(".names.json")
    names = json.loads(side.read_text()) if side.exists() else [f"b{i+1}" for i in range(arr.shape[0])]
    return arr, prof, names


def bounds_of(prof):
    t, w, h = prof["transform"], prof["width"], prof["height"]
    return (t.c, t.f + t.e * h, t.c + t.a * w, t.f)  # west, south, east, north


def save_raster(arr2d, prof, path, nodata=np.nan):
    p = prof.copy()
    p.update(count=1, dtype="float32", nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr2d.astype("float32"), 1)
    return path


def pixel_size_m(prof):
    lat = (prof["transform"].f + prof["transform"].f + prof["transform"].e * prof["height"]) / 2
    return abs(prof["transform"].a) * 111320 * np.cos(np.deg2rad(lat))


# ------------------------------------------------------------------ datasets
def _groups(rows, cols, prof, block_m=5000):
    blk = max(int(block_m / pixel_size_m(prof)), 1)
    return (rows // blk) * 100000 + (cols // blk)


def dataset_from_label_band(arr, prof, names, label_band, factors, n_per_class=6000, seed=SEED):
    """Balanced pixel sample where label_band is 1 (positive) / 0 (negative) / NaN (ignored)."""
    rng = np.random.default_rng(seed)
    F = np.stack([arr[names.index(f)] for f in factors])
    lab = arr[names.index(label_band)]
    ok = np.isfinite(F).all(axis=0) & np.isfinite(lab)
    picks = []
    for cls in (1, 0):
        r, c = np.where(ok & (lab == cls))
        if len(r) == 0:
            raise RuntimeError(f"No pixels with label {cls}")
        idx = rng.choice(len(r), size=min(n_per_class, len(r)), replace=False)
        picks.append(np.c_[r[idx], c[idx], np.full(len(idx), cls)])
    p = np.vstack(picks)
    rows, cols, y = p[:, 0].astype(int), p[:, 1].astype(int), p[:, 2].astype(int)
    X = pd.DataFrame(F[:, rows, cols].T, columns=factors)
    return X, y, _groups(rows, cols, prof), (rows, cols)


def dataset_from_points(arr, prof, names, lonlat, factors, neg_ratio=1.0, exclude_px=8, seed=SEED, extent=None,
                        block_m=5000):
    """Positive = pixels containing the given (lon, lat) points; negatives drawn away from all positives.

    extent = (west, south, east, north) of the area the inventory actually covers. Landslides outside it were never
    mapped, so unmapped cells are NOT true negatives: when given, negatives are drawn only inside this extent.
    """
    rng = np.random.default_rng(seed)
    t = prof["transform"]
    cols = ((lonlat[:, 0] - t.c) / t.a).astype(int)
    rows = ((lonlat[:, 1] - t.f) / t.e).astype(int)
    H, W = arr.shape[1:]
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols = rows[inside], cols[inside]
    F = np.stack([arr[names.index(f)] for f in factors])
    ok = np.isfinite(F).all(axis=0)
    keep = ok[rows, cols]
    rows, cols = rows[keep], cols[keep]
    rc = np.unique(np.c_[rows, cols], axis=0)                    # one sample per pixel (many mapped failures share a pixel)
    rows, cols = rc[:, 0], rc[:, 1]
    pos = np.zeros((H, W), bool)
    pos[rows, cols] = True
    near = ndimage.binary_dilation(pos, iterations=exclude_px)
    cand = ok & ~near
    if extent is not None:
        lon = t.c + t.a * (np.arange(W) + 0.5)
        lat = t.f + t.e * (np.arange(H) + 0.5)
        in_x = (lon >= extent[0]) & (lon <= extent[2])
        in_y = (lat >= extent[1]) & (lat <= extent[3])
        cand &= in_y[:, None] & in_x[None, :]
    nr, nc = np.where(cand)
    n_neg = int(len(rows) * neg_ratio)
    idx = rng.choice(len(nr), size=min(n_neg, len(nr)), replace=False)
    R = np.r_[rows, nr[idx]]
    Cc = np.r_[cols, nc[idx]]
    y = np.r_[np.ones(len(rows), int), np.zeros(len(idx), int)]
    X = pd.DataFrame(F[:, R, Cc].T, columns=factors)
    return X, y, _groups(R, Cc, prof, block_m), (R, Cc)


# ------------------------------------------------------------------ models
def make_models():
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    stack = StackingClassifier([("rf", rf), ("xgb", xgb), ("et", et)],
                               final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1)
    return {"RandomForest": rf, "XGBoost": xgb, "ExtraTrees": et, "Stacking (RF+XGB+ET)": stack}


class FRTransformer(BaseEstimator, TransformerMixin):
    """Frequency-ratio encoding (statistical bivariate method) used to build FR-ML hybrid models.

    Each factor is cut into quantile classes; every value is replaced by FR = (share of positives in the class)
    / (share of all samples in the class). Fitted on training data only, so there is no leakage.
    """

    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.cols_, self.edges_, self.fr_ = list(X.columns), {}, {}
        for c in self.cols_:
            e = np.unique(np.quantile(X[c], np.linspace(0, 1, self.bins + 1)[1:-1]))
            b = np.digitize(X[c], e)
            fr = np.ones(len(e) + 1)
            for k in range(len(e) + 1):
                m = b == k
                if m.sum() and y.sum():
                    fr[k] = (y[m].sum() / y.sum()) / (m.sum() / len(y)) if m.sum() else 1.0
            self.edges_[c], self.fr_[c] = e, fr
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols_)
        out = pd.DataFrame(index=X.index)
        for c in self.cols_:
            out[c] = self.fr_[c][np.digitize(X[c], self.edges_[c])]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_)


def spatial_split(groups, test_size=0.3, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def evaluate(models, X, y, groups):
    """Spatial hold-out + 5-fold spatial-block CV. Returns (metrics DataFrame, fitted dict, test data)."""
    tr, te = spatial_split(groups)
    rows, fitted, preds = [], {}, {}
    for name, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        fitted[name], preds[name] = m, p
        cv = []
        for a, b in GroupKFold(5).split(X, y, groups):
            mm = clone(m)
            mm.fit(X.iloc[a], y[a])
            cv.append(roc_auc_score(y[b], mm.predict_proba(X.iloc[b])[:, 1]))
        rows.append(dict(model=name, AUC_test=roc_auc_score(y[te], p), AP_test=average_precision_score(y[te], p),
                         Accuracy=accuracy_score(y[te], p > 0.5), F1=f1_score(y[te], p > 0.5),
                         AUC_5fold_mean=np.mean(cv), AUC_5fold_sd=np.std(cv)))
    return pd.DataFrame(rows).round(3), fitted, (te, preds)


def plot_roc(y_test, preds, path, title="ROC curves (spatial hold-out)"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n, p in preds.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def importance(model, X, y, path=None, n=2000, title="Permutation importance (AUC drop)"):
    idx = np.random.default_rng(SEED).choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx], y[idx], scoring="roc_auc", n_repeats=8,
                               random_state=SEED, n_jobs=1)
    df = pd.DataFrame({"factor": X.columns, "importance": r.importances_mean, "sd": r.importances_std})
    df = df.sort_values("importance")
    if path:
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(df) + 1.5))
        ax.barh(df["factor"], df["importance"], xerr=df["sd"], color="#2b6c8f")
        ax.set(xlabel="Mean decrease in AUC", title=title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return df.sort_values("importance", ascending=False).round(4)


def predict_map(model, arr, names, factors, chunk=400000):
    F = np.stack([arr[names.index(f)] for f in factors])
    H, W = F.shape[1:]
    ok = np.isfinite(F).all(axis=0)
    flat = F.reshape(len(factors), -1).T
    okf = ok.reshape(-1)
    out = np.full(H * W, np.nan, dtype="float32")
    idx = np.where(okf)[0]
    for k in range(0, len(idx), chunk):
        sel = idx[k:k + chunk]
        out[sel] = model.predict_proba(pd.DataFrame(flat[sel], columns=factors))[:, 1]
    return out.reshape(H, W)


def classify_quantiles(prob, n=5):
    """Susceptibility classes 1..n using quantile breaks (very low ... very high)."""
    v = prob[np.isfinite(prob)]
    br = np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])
    cls = np.digitize(prob, br) + 1
    cls = cls.astype("float32")
    cls[~np.isfinite(prob)] = np.nan
    return cls, br


def class_area_table(cls, prof, labels=("Very low", "Low", "Moderate", "High", "Very high")):
    px_km2 = (pixel_size_m(prof) / 1000) ** 2
    rows = []
    tot = np.isfinite(cls).sum()
    for k, lab in enumerate(labels, 1):
        n = int((cls == k).sum())
        rows.append(dict(Class=lab, Area_km2=round(n * px_km2, 1), Percent=round(100 * n / tot, 1)))
    return pd.DataFrame(rows)
