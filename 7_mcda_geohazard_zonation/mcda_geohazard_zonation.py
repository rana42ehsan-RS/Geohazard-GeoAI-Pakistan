"""7 - Multi-criteria decision analysis (MCDA) for geohazard zonation with GIS and AI.

AOI: Muzaffarabad - Neelum - Murree (Hazara arc).
Methods : AHP (expert pairwise matrix + consistency ratio), entropy weighting, weighted linear combination,
          sensitivity analysis, and an AI (random forest) model when a landslide inventory is available.
Run:  python mcda_geohazard_zonation.py
Edit the AHP matrix below to reflect your own expert judgement / literature.
"""
import ee
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import config as C, gee, ml, pipeline, viz, webdata
from common.report import Report

TAG = "a07"
FACTORS = ["slope", "rain_annual", "dist_drainage", "seismic_kde", "ndvi", "landcover", "curvature", "twi"]
BENEFIT = {"slope": +1, "rain_annual": +1, "dist_drainage": -1, "seismic_kde": +1, "ndvi": -1, "landcover": 0,
           "curvature": 0, "twi": +1}   # direction of hazard contribution (landcover handled by lookup below)
LC_SCORE = {10: 0.2, 20: 0.4, 30: 0.5, 40: 0.5, 50: 0.7, 60: 1.0, 70: 0.3, 80: 0.0, 90: 0.0, 95: 0.0, 100: 0.3}

# Saaty 1-9 pairwise comparison, rows/cols in FACTORS order (upper triangle given, reciprocals filled automatically)
UPPER = {
    ("slope", "rain_annual"): 3, ("slope", "dist_drainage"): 3, ("slope", "seismic_kde"): 2, ("slope", "ndvi"): 4,
    ("slope", "landcover"): 5, ("slope", "curvature"): 5, ("slope", "twi"): 6,
    ("rain_annual", "dist_drainage"): 1, ("rain_annual", "seismic_kde"): 1 / 2, ("rain_annual", "ndvi"): 2,
    ("rain_annual", "landcover"): 3, ("rain_annual", "curvature"): 3, ("rain_annual", "twi"): 4,
    ("dist_drainage", "seismic_kde"): 1 / 2, ("dist_drainage", "ndvi"): 2, ("dist_drainage", "landcover"): 2,
    ("dist_drainage", "curvature"): 3, ("dist_drainage", "twi"): 3,
    ("seismic_kde", "ndvi"): 3, ("seismic_kde", "landcover"): 3, ("seismic_kde", "curvature"): 4, ("seismic_kde", "twi"): 4,
    ("ndvi", "landcover"): 1, ("ndvi", "curvature"): 2, ("ndvi", "twi"): 2,
    ("landcover", "curvature"): 1, ("landcover", "twi"): 2, ("curvature", "twi"): 1,
}
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(factors, upper):
    n = len(factors)
    A = np.ones((n, n))
    for (a, b), v in upper.items():
        i, j = factors.index(a), factors.index(b)
        A[i, j], A[j, i] = v, 1 / v
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real); w /= w.sum()
    ci = (vals[k].real - n) / (n - 1)
    return w, ci / RI[n], A


def scale01(a, invert=False):
    lo, hi = np.nanpercentile(a, [2, 98])
    z = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    return 1 - z if invert else z


def normalise_layers(arr, names):
    L = {}
    for f in FACTORS:
        a = arr[names.index(f)]
        if f == "landcover":
            z = np.full(a.shape, np.nan)
            for k, v in LC_SCORE.items():
                z[a == k] = v
            L[f] = z
        elif f in ("curvature",):
            L[f] = scale01(np.abs(a))
        else:
            L[f] = scale01(a, invert=BENEFIT[f] < 0)
    return L


def wlc(L, w):
    out = np.zeros_like(next(iter(L.values())))
    for f, wi in zip(FACTORS, w):
        out += wi * np.nan_to_num(L[f], nan=0)
    valid = np.all([np.isfinite(L[f]) for f in FACTORS], axis=0)
    return np.where(valid, out, np.nan)


def entropy_w(L):
    ok = np.all([np.isfinite(L[f]) for f in FACTORS], axis=0)          # same cells for every criterion
    M = np.stack([L[f][ok][::50] for f in FACTORS], axis=1) + 1e-6
    P = M / M.sum(axis=0)
    E = -(P * np.log(P)).sum(axis=0) / np.log(len(M))
    d = 1 - E
    return d / d.sum()


def main():
    C.ensure(); gee.init()
    out = C.out_dir(TAG)
    cfg = C.SITES[TAG]
    geom = gee.region(cfg["bbox"])
    ter = gee.terrain_stack(geom).select(["slope", "dist_drainage", "curvature", "twi"])
    rain = gee.rain_mean_annual(geom)
    ndvi = gee.s2_median(geom, "2022-04-01", "2022-10-31").select("ndvi")
    lc = gee.worldcover(geom).toFloat()
    base = ee.Image.cat([ter, rain, ndvi, lc]).clip(geom)
    tif = gee.download(base, TAG, "stack", cfg["bbox"], cfg["scale"], tile_px=300)
    arr, prof, names = ml.read_stack(tif)

    # ---- seismicity kernel density from USGS (M>=3.5, 1973-2025), computed on the raster grid
    cat = webdata.usgs_catalog([cfg["bbox"][0] - 1, cfg["bbox"][1] - 1, cfg["bbox"][2] + 1, cfg["bbox"][3] + 1],
                               "1973-01-01", "2025-12-31", 3.5)
    H, W = arr.shape[1:]
    t = prof["transform"]
    lon = t.c + t.a * (np.arange(W) + 0.5)
    lat = t.f + t.e * (np.arange(H) + 0.5)
    step = 10   # evaluate KDE on a coarse grid then upsample
    LL, LA = np.meshgrid(lon[::step], lat[::step])
    dens = np.zeros_like(LL)
    for _, e in cat.iterrows():
        d = np.hypot((LL - e.longitude) * 0.84, LA - e.latitude) * 111
        dens += (10 ** (0.5 * (e.mag - 3.5))) * np.exp(-0.5 * (d / 15) ** 2)
    dens_full = np.kron(dens, np.ones((step, step)))[:H, :W]
    arr = np.concatenate([arr, dens_full[None].astype("float32")])
    names = names + ["seismic_kde"]

    L = normalise_layers(arr, names)
    w_ahp, cr, A = ahp(FACTORS, UPPER)
    w_ent = entropy_w(L)
    w_comb = 0.5 * w_ahp + 0.5 * w_ent
    S_ahp, S_ent, S_comb = wlc(L, w_ahp), wlc(L, w_ent), wlc(L, w_comb)
    b = ml.bounds_of(prof)

    wt = pd.DataFrame({"factor": FACTORS, "AHP": w_ahp, "Entropy": w_ent, "Combined": w_comb}).round(3)
    wt.to_csv(out / "weights.csv", index=False)
    for nm, S in (("AHP", S_ahp), ("Entropy", S_ent), ("Combined", S_comb)):
        c, _ = ml.classify_quantiles(S)
        ml.save_raster(S, prof, out / f"hazard_index_{nm}.tif")
        ml.save_raster(c, prof, out / f"hazard_zones_{nm}.tif")
        viz.plot_classes(c, b, f"Geohazard zonation - {nm}", out / f"zones_{nm}.png")
    cls_ahp, _ = ml.classify_quantiles(S_ahp)
    area = ml.class_area_table(cls_ahp, prof)

    # ---- sensitivity: perturb each weight +-20 % (renormalised) and measure class change
    base_c = cls_ahp
    sens = []
    for i, f in enumerate(FACTORS):
        row = {"factor": f}
        for s, lab in ((0.8, "-20%"), (1.2, "+20%")):
            w = w_ahp.copy(); w[i] *= s; w /= w.sum()
            c, _ = ml.classify_quantiles(wlc(L, w))
            ok = np.isfinite(base_c)
            row[f"changed_{lab}"] = round(100 * (c[ok] != base_c[ok]).mean(), 1)
        sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)

    # ---- agreement between methods
    ok = np.isfinite(S_ahp) & np.isfinite(S_ent)
    rho = spearmanr(S_ahp[ok][::20], S_ent[ok][::20])[0]

    r = Report(cfg["title"], cfg["site"])
    r.h("1. Introduction"); r.p(cfg["why"])
    r.h("2. Data and methods")
    r.bullets(["Criteria (8): slope, rainfall, distance to drainage, seismic kernel density (USGS), NDVI, land cover, curvature, TWI.",
               "Normalisation: robust min-max (2nd-98th percentile) with direction set by the hazard-contribution sign; land cover by an expert score lookup.",
               "AHP: pairwise comparison matrix (Saaty scale), principal eigenvector weights, consistency ratio (CR).",
               "Entropy weights derived from the data; combined weights = mean of AHP and entropy.",
               "Weighted linear combination -> quantile zones (very low ... very high). OAT sensitivity (+-20 % weight change)."])
    r.h("3. Results")
    r.p(f"AHP consistency ratio CR = {cr:.3f} ({'acceptable (<0.10)' if cr < 0.1 else 'NOT acceptable - revise the matrix'}). "
        f"Spearman correlation between AHP and entropy indices: {rho:.2f}.")
    r.table(wt, "Criteria weights")
    r.table(pd.DataFrame(A.round(2), index=FACTORS, columns=FACTORS).reset_index().rename(columns={"index": "AHP matrix"}),
            "AHP pairwise comparison matrix")
    for nm in ("AHP", "Entropy", "Combined"):
        r.fig(out / f"zones_{nm}.png", f"Zonation using {nm} weights", 5.6)
    r.table(area, "Area per zone (AHP)")
    r.table(sens, "Sensitivity: % of cells changing class")

    # ---- AI / validation with inventory
    inv = webdata.load_landslide_inventory(cfg["bbox"])
    if inv is not None and len(inv) >= 30:
        ext = webdata.inventory_extent(inv)
        # coarse layers (5 km rainfall, ~1 km earthquake density) act as location proxies inside the ~27 km mapped area,
        # so the random forest uses only fine-scale predictors; they stay in the MCDA, where they vary across the whole AOI
        AI_FACTORS = [f for f in FACTORS if f not in ("rain_annual", "seismic_kde")]
        Xp, y, groups, (rr, cc) = ml.dataset_from_points(arr, prof, names, inv, AI_FACTORS, extent=ext, block_m=3000)
        auc = {}
        for nm, S in (("AHP", S_ahp), ("Entropy", S_ent), ("Combined", S_comb)):
            v = np.isfinite(S[rr, cc])
            auc[nm] = roc_auc_score(y[v], S[rr, cc][v])
        r.h("AI model and validation against the landslide inventory", 2)
        r.p("Validation inventory: 3,820 mapped landslides (polygons reduced to centroids) from QuickBird/DigitalGlobe imagery, mostly 2006 and 2010. Non-landslide samples come only from inside the inventory's mapped extent, because unmapped ground is not a verified negative. The random forest uses fine-scale predictors only (slope, drainage, NDVI, land cover, curvature, TWI).")
        r.p("MCDA index AUCs against inventory: " + ", ".join(f"{k}={v:.3f}" for k, v in auc.items()))
        pd.DataFrame([auc]).round(3).to_csv(out / "mcda_validation_auc.csv", index=False)
        ai = pipeline.run(TAG + "_ai", arr, prof, names, Xp, y, groups, AI_FACTORS, "Geohazard", r,
                          points=(inv[:, 0], inv[:, 1]))
    else:
        r.p("No landslide inventory was available, so the MCDA zones are NOT validated. Place a point inventory in "
            "data/inventories/ and re-run to obtain AUC validation and the random-forest comparison.")
    r.h("4. Limitations")
    r.p("MCDA weights embed subjective judgement; lithology and fault distance are omitted because no open consistent layer was used. "
        "Quantile classes are relative to this AOI and are not absolute hazard probabilities.")
    r.references(["ahp", "gee", "chirps", "rf", "worldcover"])
    r.save(out / "A07_report.docx")
    print("Done ->", out)


if __name__ == "__main__":
    main()
