"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a07': {'title': 'MCDA Geohazard Zonation with GIS and AI',
         'site': 'Muzaffarabad - Neelum - Murree (Hazara arc)',
         'bbox': [72.8, 33.5, 74.3, 34.9],
         'scale': 90,
         'why': '2005 earthquake zone with landslides, active faulting and heavy monsoon rain: multiple '
                'hazards overlap.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

