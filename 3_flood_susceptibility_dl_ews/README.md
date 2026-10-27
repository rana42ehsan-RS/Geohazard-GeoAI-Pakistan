# 3 - Deep learning + GIS flood susceptibility and early warning

**Study site:** Lower Indus: Dadu, Jamshoro, Larkana (Sindh)

## Files

| File | Purpose |
|---|---|
| `flood_susceptibility_dl_ews.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, ml, pipeline, references, report, viz, webdata) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` (data files are not part of this repository).
4. Run: `python flood_susceptibility_dl_ews.py`
5. Results are written to `outputs/`.
