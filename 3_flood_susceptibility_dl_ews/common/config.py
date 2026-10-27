"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a03': {'title': 'Deep Learning + GIS for Flood Susceptibility and Early Warning',
         'site': 'Lower Indus: Dadu, Jamshoro, Larkana (Sindh)',
         'bbox': [66.8, 26.0, 68.8, 27.8],
         'scale': 200,
         'why': 'Worst-hit area of the 2022 floods; clear Sentinel-1 flood labels and flat terrain for CNN '
                'learning.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

