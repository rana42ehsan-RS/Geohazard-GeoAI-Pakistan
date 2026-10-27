"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a05': {'title': 'LULC Change and Natural Hazard Dynamics (multi-decadal)',
         'site': 'Swat valley, Khyber Pakhtunkhwa',
         'bbox': [71.8, 34.4, 72.9, 35.7],
         'scale': 90,
         'why': 'Documented deforestation, rapid built-up growth in the floodplain and severe floods in 2010 '
                'and 2022.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

