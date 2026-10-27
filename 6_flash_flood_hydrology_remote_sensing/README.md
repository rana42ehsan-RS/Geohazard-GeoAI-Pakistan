# 6 - Rainfall-runoff model + remote sensing for flash floods

**Study site:** Zhob River catchment, Balochistan (data-scarce, ungauged)

## Files

| File | Purpose |
|---|---|
| `flash_flood_hydrology_remote_sensing.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, ml, references, report, viz, webdata) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` (data files are not part of this repository).
4. Run: `python flash_flood_hydrology_remote_sensing.py`
5. Results are written to `outputs/`.
