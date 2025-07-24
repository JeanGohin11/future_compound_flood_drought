import netCDF4
import xarray as xr
import os
from dask.diagnostics import ProgressBar  

"""
This script was used to retrieve and merge files from lustre on Anunna, and take the medians between models
"""

#list of models
models = ["gfdl-esm4",  "ipsl-cm6a-lr",  "mpi-esm1-2-hr", "mri-esm2-0",  "ukesm1-0-ll"]
# list of scenarios
scenarios = ["ssp126", "ssp370"]
# List of file periods
periods = ["2051_2060", "2061_2070", "2071_2080", "2081_2090", "2091_2100"]

models_data_ssp126 = {}
models_data_ssp370 = {}
models_data_ssp585 = {}
# Load datasets for each model and scenario
for model in models:
    data_dir = f"/lustre/backup/WUR/ESG/data/PROJECT_DATA/ISIMIP3b/water_global/vic-wur/20230911/future/{model}/"
    file_paths1 = []
    file_paths2 = []
    file_paths3 = []
    # Load datasets for each period
    for period in periods:
        path1 = os.path.join(data_dir, f"vic-wur_{model}_w5e5_ssp126_nat_default_dis_global_daily_{period}.nc")
        path2 = os.path.join(data_dir, f"vic-wur_{model}_w5e5_ssp370_nat_default_dis_global_daily_{period}.nc")
        path3 = os.path.join(data_dir, f"vic-wur_{model}_w5e5_ssp585_nat_default_dis_global_daily_{period}.nc")
        file_paths1.append(path1)
        file_paths2.append(path2)
        file_paths3.append(path3)
    models_data_ssp126[model] = xr.open_mfdataset(file_paths1, concat_dim="time", combine="nested")
    models_data_ssp370[model] = xr.open_mfdataset(file_paths2, concat_dim="time", combine="nested")
    models_data_ssp585[model] = xr.open_mfdataset(file_paths3, concat_dim="time", combine="nested")

dis_arrays_ssp126 = []
dis_arrays_ssp370 = []
dis_arrays_ssp585 = []

for model, dataset in models_data_ssp126.items():
        dis_arrays_ssp126.append(dataset["dis"])

for model, dataset in models_data_ssp370.items():
        dis_arrays_ssp370.append(dataset["dis"])

for model, dataset in models_data_ssp585.items():
        dis_arrays_ssp585.append(dataset["dis"])

# Combine all 'dis' arrays into a single DataArray with a new dimension 'model'
dis_combined_ssp126 = xr.concat(dis_arrays_ssp126, dim="model")
dis_combined_ssp370 = xr.concat(dis_arrays_ssp370, dim="model")
dis_combined_ssp585 = xr.concat(dis_arrays_ssp585, dim="model")
# Compute the median across models


dis_combined_ssp126 = dis_combined_ssp126.chunk(chunks= {"time": -1, "lon": 200, "lat": 200})
dis_combined_ssp370 = dis_combined_ssp370.chunk(chunks={"time": -1, "lon": 200, "lat": 200})
dis_combined_ssp585 = dis_combined_ssp585.chunk(chunks={"time": -1, "lon": 200, "lat": 200})
# Compute the median across models
model_median_ssp126= dis_combined_ssp126.median(dim="model", skipna = True)
model_median_ssp370= dis_combined_ssp370.median(dim="model", skipna = True)
model_median_ssp585= dis_combined_ssp585.median(dim="model", skipna = True)

# Save the merged dataset to a new NetCDF file
with ProgressBar():
    # model_median_ssp126.to_netcdf("Data/historical_model_median_dis_ssp126_2051_2100.nc")
    # model_median_ssp370.to_netcdf("Data/historical_model_median_dis_ssp370_2051_2100.nc")
    model_median_ssp585.to_netcdf("Data/model_median_dis_ssp585_2051_2100.nc") #save the merged dataset to a new NetCDF file



