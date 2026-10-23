# Example code for "Machine-learning-based approach for solar radiation model prediction error identification, attribution and correction"

This folder contains **example Python scripts** showing how the XGBoost residual
(error) models and the SHAP analysis for the CAMS Radiation Service (CRS) GHI,
DHI and BNI products of the manuscript were run.

**Purpose.** The scripts are shared to illustrate how each step of the analysis
is implemented and how the steps fit together, as a basis for your own
implementation. They are *not* a ready-to-run reproduction package: the
collocated CRS/ground-observation data are not shared here, and the scripts were written at different stages
of the work. Expect to adapt paths,
file layouts and feature lists to your own data.

Codes licensed under the MIT License. Copyright 2026 Antti Lipponen / Finnish Meteorological Institute.

## Method in brief

* Target: residual `e = obs - CRS` for each component (GHI, DHI, BNI), in W/m².
* Predictors: CRS input variables (cloud optical depth/type/coverage, aerosol
  optical depths by species, BRDF kernels `fgeo/fiso/fvol`, albedo, `tcwv`,
  `tco3`, SZA, season indicator, snow probability).
* Model: XGBoost (`reg:squarederror`). Corrected irradiance = CRS + predicted residual.
* Attribution: TreeSHAP (Python `shap` package), grouped into cloud, aerosol,
  surface, gas and other features.
* Two model families: *diagnostic* models (in-sample, used for the SHAP analysis)
  and separately tuned *predictive correction* models (station-day test split,
  leave-one-station-out sensitivity).

## Scripts

| Script | What it shows |
|---|---|
| `run_diagnostic_shap_workflow_GHI.py` | End-to-end example for **GHI**: builds `CAMS_dataset.h5` from the CRS time-series CSV files and `obs_dataset.h5` from level-1 station NetCDF files (15-min centred mean, collocated to MSG acquisition times), applies station-quality (classes `v`, `g`) and data filters (SZA < 80°, positive irradiances, valid cloud/snow fields), then fits the in-sample *diagnostic* model (100,000 sampled rows, `max_depth=15`, `colsample_bytree=0.3`, `learning_rate=0.9`, 100 rounds) for ten random seeds (IDs 01–10), computes TreeSHAP values and makes beeswarm, dependence and waterfall plots. Paths are relative to the script (`CAMEOglobaldata/...`, `data/`, `model/`, `figs/`). The MSG pixel geometry and acquisition-time helpers (`get_pxl_data_new`, `get_pxl_time`) are included in the script. |
| `run_xgboost_grid_search.py` | *Predictive correction models*: loads the two HDF5 datasets, builds the station-day split (70/15/15 %, seed 20260901, saved to `station_date_splits.csv`) and runs a resumable grid search of 1,350 hyperparameter combinations per component (learning rate, depth, min. child weight, row/column subsampling, three regularisation profiles; up to 3,000 rounds, early stopping after 100 rounds on validation RMSE). Writes to `grid_search_results/`. |
| `run_final_xgboost_models.py` | Refits the selected hyperparameters on train+validation and evaluates once on the untouched test split (`final_model_results/`): metrics, per-station RMSE reduction (cf. Table 5), bootstrap intervals and density plots (cf. Fig. 5), station-mean-residual baseline figure. With `--leave-one-station-out` it runs the LOSO analysis (cf. Sect. 3.6, Table B1) into `leave_one_station_out_results/`. |
| `run_final_shap_analysis.py` | TreeSHAP for the final correction models on their held-out test rows (the sensitivity analysis of Sect. 2.5): saves SHAP values, feature and group importances, beeswarm plots with group symbols (cf. Figs. 2, A1, A3), SZA-binned group contributions (cf. Figs. 3, A2, A4) and the two single-sample waterfalls (cf. Fig. 4; Palaiseau 2021-11-14 09:26 UTC, Cabauw 2021-08-05 17:11 UTC). |

The three `run_*` scripts after the workflow script import each other and
must stay in the same folder (`run_final_*` import `run_xgboost_grid_search.py`).
They share the options `--data-root DIR --cases GHI DHI BNI --nthread N`.

Typical sequence:

```bash
python run_diagnostic_shap_workflow_GHI.py          # data preparation + diagnostic SHAP (GHI)
python run_xgboost_grid_search.py --data-root /path/to/data
python run_final_xgboost_models.py --data-root /path/to/data
python run_final_xgboost_models.py --data-root /path/to/data --leave-one-station-out
python run_final_shap_analysis.py --data-root /path/to/data
```

Requirements: Python 3.10+, `numpy`, `pandas`, `pytables`, `xarray`, `netCDF4`,
`xgboost`, `shap`, `matplotlib`, `joblib`.

## Data (not included here)

`run_diagnostic_shap_workflow_GHI.py` creates `data/CAMS_dataset.h5` and
`data/obs_dataset.h5`. The grid-search/final/SHAP scripts read the same two
tables from `DATA_ROOT/<CASE>data/` (`GHIdata`, `DHIdata`, `BNIdata`), so copy
or link the files into each of these folders. Both tables are indexed by
`(ObservationPeriod, station)`:

* `CAMS_dataset.h5`: CRS irradiances (`GHI`, `DHI`, `BNI`), `TOA`, and the predictors
  listed in `FEATURES` in `run_xgboost_grid_search.py`
  (`sza, summerWinterSplit, tco3, tcwv, AODBC, AODDU, AODSS, AODOR, AODSU, AODNI,
  AODAM, AODSO, Snow_probability, fiso, fvol, fgeo, albedo, CloudOpticalDepth,
  CloudCoverage, CloudType`).
* `obs_dataset.h5`: ground observations (`GHI`, `DHI`, `BNI`) averaged over the
  15-min window centred on the satellite observation time.

Station IDs have the form `BSRN_PAL`, `SAURAN_CSIR`, `enerMENA_ZAG`. CRS data are
available from the CAMS Atmosphere Data Store; see "Code and data availability"
in the manuscript for the ground observations.
