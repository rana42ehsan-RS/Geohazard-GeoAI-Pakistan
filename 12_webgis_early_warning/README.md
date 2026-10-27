# 12 - WebGIS early-warning prototype with live feeds

**Study site:** Pakistan (national), with snapshot layers from other study sites

## Files

| File | Purpose |
|---|---|
| `webgis_early_warning.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, ml, references, report, viz, webdata) |
| `streamlit_app.py` | optional interactive dashboard |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` and result layers (GeoJSON/GeoTIFF) in `data/layers/` (data files are not part of this repository).
4. Run: `python webgis_early_warning.py`  (dashboard: `streamlit run streamlit_app.py`)
5. Results are written to `outputs/`.
