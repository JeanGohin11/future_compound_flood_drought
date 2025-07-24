import netCDF4
import xarray as xr
import os
from dask.diagnostics import ProgressBar  
"""This file imports and merges soil moisture files from future scenarios and takes the median across
the three soil layers from VIC-WUR"""

#list of models
models = ["gfdl-esm4",  "ipsl-cm6a-lr",  "mpi-esm1-2-hr", "mri-esm2-0",  "ukesm1-0-ll"]
# List of file periods
periods = ["2051_2060", "2061_2070", "2071_2080", "2081_2090", "2091_2100"]

scenarios = ["ssp126", "ssp370", "ssp585"]

scenarios_data = {}
for scenario in scenarios:
    models_data = {}
    for model in models:
        data_dir = f"/lustre/backup/WUR/ESG/data/PROJECT_DATA/ISIMIP3b/water_global/vic-wur/20230911/future/{model}/"
        file_paths = []
        for period in periods:
            path = os.path.join(data_dir, f"vic-wur_{model}_w5e5_{scenario}_nat_default_soilmoist_global_daily_{period}.nc")
            file_paths.append(path)
        models_data[model] = xr.open_mfdataset(file_paths, concat_dim="time", combine="nested")
    scenarios_data[scenario] = models_data

soilmoist_arrays_scenarios = {}

for scenario in scenarios_data:
    soilmoist_arrays = []
    for dataset in models_data.values():
        soilmoist_median = dataset['soilmoist'].median(dim=['layer'], skipna = True)
        soilmoist_arrays.append(soilmoist_median)
    soilmoist_arrays_scenarios[scenario] = soilmoist_arrays


soilmoist_combined = xr.concat(soilmoist_arrays_scenarios["ssp585"], dim="model")
soilmoist_median = soilmoist_combined.median(dim="model", skipna = True)
with ProgressBar():
    soilmoist_median.to_netcdf(f"Data/historical_model_median_soilmoist_ssp585_2051_2100.nc")   