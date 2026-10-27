# 11 - Forest-fire susceptibility and patrol-hub placement

**Study site:** Margalla Hills - Murree - Galiyat - Abbottabad

## Files

| File | Purpose |
|---|---|
| `forest_fire_geoai.py` | the analysis script |
| `common/` | helper code used by the script (boundaries, config, gee, ml, pipeline, references, report, viz) |
| `environment.yml` | Python packages |

## Quick start

1. Create the environment: `conda env create -f environment.yml`, then `conda activate geohazard`.
2. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
3. Add the reference layers `data/boundaries/Pak_Districts.shp` and `data/boundaries/Kashmir.shp` (data files are not part of this repository).
4. Run: `python forest_fire_geoai.py`
5. Results are written to `outputs/`.
