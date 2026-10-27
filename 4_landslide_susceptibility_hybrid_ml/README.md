# 4 - Landslide susceptibility with hybrid machine learning

**Study site:** Muzaffarabad - Balakot area (Hazara arc), covered by the supplied landslide inventory

## Files

| File | Purpose |
|---|---|
| `landslide_susceptibility_hybrid_ml.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, ml, pipeline, references, report, viz, webdata) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` and the landslide inventory in `data/inventories/` (data files are not part of this repository).
4. Run: `python landslide_susceptibility_hybrid_ml.py`
5. Results are written to `outputs/`.
