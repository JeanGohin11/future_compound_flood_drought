import netCDF4
import xarray as xr
import os
"""This file imports and merges discharge data from the historical periods, takes the median across 5 GCM outputs, and saves into
a new file"""
#List of models
models = ["gfdl-esm4",  "ipsl-cm6a-lr",  "mpi-esm1-2-hr", "mri-esm2-0",  "ukesm1-0-ll"]

# List of file periods
year_ranges = ["1961_1970", "1971_1980", "1981_1990", "1991_2000", "2001_2010", "2011_2014"]

models_data = {}
for model in models:
    file_paths = []
    data_dir = f"/lustre/backup/WUR/ESG/data/PROJECT_DATA/ISIMIP3b/water_global/vic-wur/20230911/historical/{model}/"
    for years in year_ranges:
        path = os.path.join(data_dir, f"vic-wur_{model}_w5e5_historical_nat_default_dis_global_daily_{years}.nc")
        file_paths.append(path)
    for path in file_paths:
        ds = xr.open_dataset(path)
        print(ds)
    models_data[model] = xr.open_mfdataset(file_paths, concat_dim="time", combine="nested")

dis_arrays = []



for model, dataset in models_data.items():
    dis_arrays.append(dataset["dis"])

# Combine all 'dis' arrays into a single DataArray with a new dimension 'model'
dis_combined = xr.concat(dis_arrays, dim="model")

# Compute the median across models
full_model_median = dis_combined.median(dim="model", skipna=True)
model_median = full_model_median.sel(time=slice("1965-01-01", "2014-12-31"))

# Save the merged dataset to a new NetCDF file
model_median.to_netcdf("historical_model_median_1964_2014.nc")