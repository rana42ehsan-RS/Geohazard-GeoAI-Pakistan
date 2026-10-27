"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a06': {'title': 'Hydrological Modeling + Remote Sensing for Flash Flood Prediction',
         'site': 'Zhob River catchment, Balochistan (data-scarce, ungauged)',
         'bbox': [67.3, 30.2, 70.1, 32.2],
         'scale': 150,
         'why': 'Semi-arid mountainous catchment with sparse gauging and recurrent flash floods.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

