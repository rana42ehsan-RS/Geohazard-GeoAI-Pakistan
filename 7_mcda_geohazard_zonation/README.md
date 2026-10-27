# 7 - AHP / entropy MCDA geohazard zonation with an AI comparison

**Study site:** Muzaffarabad - Neelum - Murree (Hazara arc)

## Files

| File | Purpose |
|---|---|
| `mcda_geohazard_zonation.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, ml, pipeline, references, report, viz, webdata) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` and the landslide inventory in `data/inventories/` (data files are not part of this repository).
4. Run: `python mcda_geohazard_zonation.py`
5. Results are written to `outputs/`.
