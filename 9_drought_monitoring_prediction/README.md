# 9 - Multi-index drought fusion and forecasting

**Study site:** Balochistan province (district level)

## Files

| File | Purpose |
|---|---|
| `drought_monitoring_prediction.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, references, report, viz) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` (data files are not part of this repository).
4. Run: `python drought_monitoring_prediction.py`
5. Results are written to `outputs/`.
