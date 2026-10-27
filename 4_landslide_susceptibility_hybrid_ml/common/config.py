"""Central configuration: paths, GEE project and the study site of this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GEE_PROJECT = os.getenv("GEE_PROJECT", "")  # set your Google Cloud project id (see README)

# bbox = [west, south, east, north] in EPSG:4326
SITES = {'a04': {'title': 'Landslide Susceptibility with Hybrid ML and Terrain Analysis',
         'site': 'Muzaffarabad - Balakot area (Hazara arc), covered by the supplied landslide inventory',
         'bbox': [73.2, 34.46, 73.55, 34.79],
         'scale': 30,
         'why': 'Steep, seismically active terrain hit by the 2005 earthquake and later monsoon failures; a '
                'mapped inventory of 3,820 landslides (QuickBird/DigitalGlobe imagery, mostly 2006 and 2010) '
                'allows supervised, validated modelling.'}}


def out_dir(tag: str) -> Path:
    p = OUT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure():
    for p in (DATA / "rasters", DATA / "tables", DATA / "inventories", OUT):
        p.mkdir(parents=True, exist_ok=True)

