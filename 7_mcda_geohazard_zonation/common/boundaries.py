"""Reference boundaries drawn on every map: district outlines + labels, province names, and the Kashmir layer.

Layers (data/boundaries/):
  Pak_Districts.shp  - 167 admin-2 districts of Pakistan (incl. AJK and Gilgit-Baltistan), EPSG:4326
  Kashmir.shp        - the Indian-held part of Jammu & Kashmir, east of the Line of Control, shown on all maps and
                       labelled "Indian Illegally Occupied Jammu & Kashmir" (the author's cartographic convention).
"""
import geopandas as gpd
import matplotlib.patheffects as pe
from shapely.geometry import box

from . import config as C

BDIR = C.DATA / "boundaries"
KASHMIR_LABEL = "Indian Illegally Occupied\nJammu & Kashmir"
KASHMIR_LABEL_ONE_LINE = "Indian Illegally Occupied Jammu & Kashmir"
_cache = {}


def districts():
    if "d" not in _cache:
        g = gpd.read_file(BDIR / "Pak_Districts.shp").to_crs(4326)
        g["name"] = g["admin2Name"].fillna("").str.strip()
        _cache["d"] = g
    return _cache["d"]


def provinces():
    if "p" not in _cache:
        _cache["p"] = districts().dissolve(by="admin1Name", as_index=False)[["admin1Name", "geometry"]]
    return _cache["p"]


def kashmir():
    if "k" not in _cache:
        _cache["k"] = gpd.read_file(BDIR / "Kashmir.shp").to_crs(4326)
    return _cache["k"]


def _halo(color="white", w=2.2):
    return [pe.withStroke(linewidth=w, foreground=color)]


def add_context(ax, bounds, label_districts=True, max_district_labels=40, fontsize=None):
    """Draw district outlines, labels for the districts in/around the extent, and the labelled Kashmir polygon.

    bounds = (west, south, east, north). Keeps the current axis limits (set them before or after; they are restored).
    """
    w, s, e, n = bounds
    span = max(e - w, n - s)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if span > 12:                         # national maps: widen the view so the whole Kashmir polygon is visible
        kb = kashmir().total_bounds
        w, s, e, n = min(w, kb[0]), min(s, kb[1]), max(e, kb[2] + 0.1), max(n, kb[3] + 0.1)
        xlim, ylim = (w, e), (s, n)
    frame = box(w, s, e, n)

    d = districts()
    sub = d[d.intersects(frame)]
    if len(sub):
        sub.boundary.plot(ax=ax, color="#4d4d4d", linewidth=0.35 if span > 8 else 0.6, zorder=6)
        p = provinces()
        p[p.intersects(frame)].boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.0 if span > 8 else 1.3, zorder=7)

    fs = fontsize or (5.5 if span > 3 else 7.5)
    if label_districts and len(sub):
        if len(sub) <= max_district_labels:
            for _, r in sub.iterrows():                       # label at a point inside the VISIBLE part of each district
                vis = r.geometry.intersection(frame)
                if vis.is_empty or vis.area < 0.004 * frame.area:
                    continue
                pt = vis.representative_point()
                ax.text(pt.x, pt.y, r["name"], fontsize=fs, ha="center", va="center", color="#111", zorder=9,
                        style="italic", path_effects=_halo())
        else:                                                   # national / large-region maps: province names instead
            for _, r in provinces().iterrows():
                vis = r.geometry.intersection(frame)
                if vis.is_empty or vis.area < 0.02 * frame.area:
                    continue
                pt = vis.representative_point()
                ax.text(pt.x, pt.y, r["admin1Name"].title(), fontsize=max(fs + 1.5, 7), ha="center", va="center",
                        color="#222", weight="bold", zorder=9, path_effects=_halo(), alpha=0.85)

    k = kashmir()
    kc = k.geometry.intersection(frame)
    kc = kc[~kc.is_empty]
    if len(kc):
        gpd.GeoSeries(kc, crs=4326).plot(ax=ax, facecolor="#e6dccb", edgecolor="#8c2d04", hatch="////", linewidth=0.9,
                                         alpha=0.55, zorder=8)
        big = kc.iloc[kc.area.argmax()]
        pt = big.representative_point()
        ax.text(pt.x, pt.y, KASHMIR_LABEL, fontsize=max(fs + 0.5, 6.5 if span > 3 else 7),
                ha="center", va="center", color="#7f0000", weight="bold", style="italic", zorder=10,
                path_effects=_halo("white", 2.6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
