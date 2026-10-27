"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a01': {'title': 'AI-Driven Multi-Hazard Risk Assessment',
         'site': 'Pakistan, national scale, 167 districts (Pak_Districts, incl. AJK and Gilgit-Baltistan)',
         'bbox': [60.8, 23.6, 77.9, 37.1],
         'scale': 1000,
         'why': 'Pakistan is exposed to floods, drought, heat, earthquakes and landslides at once; districts '
                'are the unit at which NDMA/PDMA plan and allocate resources.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

