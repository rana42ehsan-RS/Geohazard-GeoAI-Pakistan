# Geohazard-GeoAI-Pakistan

| Folder | Topic | Study site |
|---|---|---|
| `1_multihazard_risk` | AI multi-hazard risk assessment (flood, drought, heat, landslide, earthquake) | Pakistan, 167 districts |
| `2_spatiotemporal_climate_geohazards` | Climate trends and machine learning for floodplain water | Indus basin |
| `3_flood_susceptibility_dl_ews` | Deep learning + GIS flood susceptibility and early warning | Lower Indus (Sindh) |
| `4_landslide_susceptibility_hybrid_ml` | Landslide susceptibility with hybrid machine learning | Muzaffarabad - Balakot |
| `5_lulc_change_hazard_dynamics` | Land-cover change 2000-2020 and runoff response | Swat valley |
| `6_flash_flood_hydrology_remote_sensing` | Rainfall-runoff model + remote sensing for flash floods | Zhob catchment (Balochistan) |
| `7_mcda_geohazard_zonation` | AHP / entropy MCDA geohazard zonation with an AI comparison | Muzaffarabad - Neelum - Murree |
| `8_seismic_vulnerability_spatial_ml` | Seismic hazard and vulnerability with spatial statistics | Pakistan seismic belt |
| `9_drought_monitoring_prediction` | Multi-index drought fusion and forecasting | Balochistan |
| `10_urban_heat_island_lahore` | Urban heat island, hot spots, drivers and risk | Lahore |
| `11_forest_fire_geoai` | Forest-fire susceptibility and patrol-hub placement | Margalla - Murree - Galiyat |
| `12_webgis_early_warning` | WebGIS early-warning prototype with live feeds | Pakistan |

## Using a project

1. Open its folder and read its `README.md`.
2. Install the packages: `conda env create -f environment.yml`.
3. Sign in to Google Earth Engine once (`earthengine authenticate`) and set `GEE_PROJECT` to your Google Cloud project id.
4. Add the reference layers listed in the folder README (data files are not part of this repository).
5. Run the script named in the folder README; results are written to `outputs/`.
