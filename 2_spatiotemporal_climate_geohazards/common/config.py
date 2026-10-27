"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a02': {'title': 'Spatiotemporal Modeling of Climate-Induced Geohazards',
         'site': 'Indus basin (upper catchment + Punjab/Sindh floodplain)',
         'bbox': [66.0, 24.0, 77.5, 37.0],
         'scale': 5000,
         'why': 'Monsoon rain, glacier/snow melt and temperature extremes drive floods along the whole '
                'Indus; 2010 and 2022 give two extreme events to learn from.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

