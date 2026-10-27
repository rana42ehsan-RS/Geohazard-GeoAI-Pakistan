"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a12': {'title': 'WebGIS Multi-Hazard Early Warning System',
         'site': 'Pakistan (national), with snapshot layers from other study sites',
         'bbox': [60.8, 23.6, 77.9, 37.1],
         'scale': 1000,
         'why': 'Integrates the outputs of all other assignments into one web interface with live feeds.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

