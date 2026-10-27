"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a11': {'title': 'GeoAI Forest Fire Susceptibility and Risk Mitigation',
         'site': 'Margalla Hills - Murree - Galiyat - Abbottabad',
         'bbox': [72.7, 33.5, 73.6, 34.3],
         'scale': 30,
         'why': 'Chir pine forest with a documented annual fire season (Apr-Jun) next to Islamabad.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

