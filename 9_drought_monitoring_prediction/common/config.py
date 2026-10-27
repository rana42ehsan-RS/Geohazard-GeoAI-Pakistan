"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a09': {'title': 'Satellite Drought Monitoring and Prediction (multi-index fusion + AI)',
         'site': 'Balochistan province (district level)',
         'bbox': [60.8, 24.8, 70.5, 32.2],
         'scale': 5000,
         'why': 'Most drought-prone, most arid province; recurrent multi-year droughts.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

