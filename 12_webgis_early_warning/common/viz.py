"""Map and chart helpers (static matplotlib + folium overlays)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from . import boundaries

CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]


def _decorate(ax, bounds, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25, lw=0.4)
    # north arrow
    ax.annotate("N", xy=(0.95, 0.93), xytext=(0.95, 0.80), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k"), ha="center", fontsize=11, fontweight="bold")


def plot_raster(arr, bounds, title, path, cmap="viridis", label="", vmin=None, vmax=None, overlay=None,
                points=None, point_label="Events"):
    """arr 2D; bounds=(w,s,e,n); overlay = GeoDataFrame boundary; points = (lon[], lat[])."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(arr, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if overlay is not None:
        overlay.boundary.plot(ax=ax, color="k", lw=0.5)
    if points is not None:
        ax.scatter(points[0], points[1], s=6, c="k", marker="x", lw=0.5, label=point_label)
        ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    boundaries.add_context(ax, bounds)
    _decorate(ax, bounds, title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_classes(cls, bounds, title, path, overlay=None, points=None, point_label="Events", labels=CLASS_LABELS,
                 colors=CLASS_COLORS):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = ListedColormap(colors[:len(labels)])
    ax.imshow(cls, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), cmap=cmap, vmin=0.5,
              vmax=len(labels) + 0.5, interpolation="nearest")
    if overlay is not None:
        overlay.boundary.plot(ax=ax, color="k", lw=0.5)
    if points is not None:
        ax.scatter(points[0], points[1], s=8, c="k", marker="x", lw=0.6, label=point_label)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors[:len(labels)]]
    ax.legend(handles + ([ax.collections[-1]] if points is not None else []),
              list(labels) + ([point_label] if points is not None else []),
              loc="lower left", fontsize=8, title="Class")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    boundaries.add_context(ax, bounds)
    _decorate(ax, bounds, title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def choropleth(gdf, col, title, path, cmap="YlOrRd", label="", scheme="quantiles", k=5):
    fig, ax = plt.subplots(figsize=(7.5, 8))
    kw = dict(categorical=True) if scheme == "unique_values" else dict(scheme=scheme.replace("_", ""), k=k)
    gdf.plot(column=col, cmap=cmap, legend=True, ax=ax, edgecolor="grey", linewidth=0.2,
             legend_kwds=dict(fontsize=8, title=str(label or col), loc="lower right"),
             missing_kwds=dict(color="lightgrey"), **kw)
    b = tuple(gdf.total_bounds)
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    boundaries.add_context(ax, b)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def raster_to_rgba(arr, cmap="viridis", vmin=None, vmax=None, alpha=0.75):
    a = np.array(arr, dtype="float32")
    vmin = np.nanmin(a) if vmin is None else vmin
    vmax = np.nanmax(a) if vmax is None else vmax
    norm = np.clip((a - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = matplotlib.colormaps[cmap](norm)
    rgba[..., 3] = np.where(np.isfinite(a), alpha, 0)
    return rgba
