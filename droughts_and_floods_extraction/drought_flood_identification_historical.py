""" This file defined thresholds for droughts and floods based on the historical distributions of discharge and soil moisture.
The drought and flood events are then detected and stored in a new file"""


import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar  
from scipy.ndimage import label

def remove_short_droughts(drought_array, min_duration=30):
    """
    Remove drought events that last less than `min_duration` days.
    
    Parameters:
    - drought_array (xr.DataArray): Boolean array where 1 represents drought, NaN otherwise.
    - min_duration (int): Minimum duration a drought must last to be kept.
    
    Returns:
    - xr.DataArray: Drought array with short events removed.
    """
    # Convert to numpy array for processing
    drought_array = xr.DataArray(drought_array)
    drought_np = np.where(drought_array == 1, 1, 0)  # Convert NaN to 0
    
    # Label connected drought events
    labeled_array, _, = label(drought_np)
    
    # Count the duration of each drought event
    event_sizes = np.bincount(labeled_array.ravel())  # Count occurrences of each label
    
    # Identify labels corresponding to short events
    remove_labels = np.where(event_sizes < min_duration)[0]  # Labels with duration < 30
    
    # Mask short droughts
    filtered_drought_np = np.where(np.isin(labeled_array, remove_labels), np.nan, drought_np)
    filtered_drought_np = np.where(filtered_drought_np == 0, np.nan, filtered_drought_np)
    
    # Convert back to xarray DataArray
    return xr.DataArray(filtered_drought_np, dims=drought_array.dims, coords=drought_array.coords)

chunk_scheme = {"time":-1, "lat": 200, "lon": 200}

# Open and merge files
historical_soilmoist = xr.open_dataset("Data/historical_model_median_soilmoist_1965_2014.nc")
historical_dis = xr.open_dataset("Data/historical_model_median_dis_1965_2014.nc")
historical = xr.merge([historical_soilmoist, historical_dis])

historical = historical.chunk(chunk_scheme)

# filter out grid cells with discharge > 100 m3/s
mask_high_discharge = (historical['dis'].groupby("time.year").mean(dim="time", skipna=True).mean(dim="year") > 100)

for variable in historical.data_vars:
    historical[variable] = xr.where(mask_high_discharge, historical[variable], np.nan)


# Compute soil moisture drought thresholds and identify soil moisture droughts
soilmoist = historical["soilmoist"]
historical = historical.drop_vars("soilmoist")

historical['thresh_soilmoist'] = soilmoist.groupby("time.month").quantile(0.15, dim="time", skipna=True).astype(np.float32)

historical['soil_drought'] = xr.where(
    soilmoist <= historical['thresh_soilmoist'].sel(month=historical["time"].dt.month),
    np.float32(1), np.nan)

historical["soil_drought"] = historical["soil_drought"].chunk({"time": -1, "lat": 200, "lon": 200})

historical["soilmoist_deficit"] =(
    soilmoist - historical['thresh_soilmoist'].sel(month=historical["time"].dt.month).drop_vars("month")).where(historical['soil_drought'] == 1).astype(np.float32)
historical["soilmoist_deficit"] = historical["soilmoist_deficit"].chunk({"time": -1, "lat": 200, "lon": 200})
historical["thresh_soilmoist"] = historical["thresh_soilmoist"].chunk({"lat": 200, "lon": 200, "month": -1})


#compute flood thresholds and floods
historical['flood_threshold'] = historical['dis'].quantile(0.95, dim="time", skipna=True).astype(np.float32)
historical['flood_threshold'] = historical['flood_threshold'].chunk({"lat":200, "lon": 200})
historical['dis_extremes'] = xr.where(historical['dis'] >= historical['flood_threshold'], np.float32(2), np.nan)
historical["volume_deficit"] = (historical['dis'] - historical['flood_threshold']).where(historical['dis_extremes'] == 2).astype(np.float32)


# Compute hydrological droughts
historical["monthly_thresholds"]  = historical["dis"].groupby("time.month").quantile(0.15, dim="time", skipna=True).astype(np.float32) 

historical["dis_extremes"] = xr.where(
    (historical["dis"] <= historical["monthly_thresholds"].sel(month=historical["time"].dt.month)),
    np.float32(1),
    historical["dis_extremes"]
).drop_vars("month")

historical["dis_extremes"] = historical["dis_extremes"].chunk(chunk_scheme)

historical["volume_deficit"] = xr.where(
    historical["dis_extremes"] == 1,
    (historical["dis"] - historical["monthly_thresholds"].sel(month=historical["time"].dt.month)).astype(np.float32).drop_vars("month"),
    historical["volume_deficit"] 
)

historical["monthly_thresholds"] = historical["monthly_thresholds"].chunk({"lat": 200, "lon": 200, "month": -1})

#merge soil drought and hydrological droughts
historical["drought"] = xr.where(
    (historical["dis_extremes"] == 1) | (historical["soil_drought"] == 1), np.float32(1), np.nan)

historical["drought"] = historical["drought"].chunk({"time": -1, "lat": 200, "lon": 200}) # rechunk drought variable

#remove droughts shorter than 30 days
historical["drought"] = xr.where(historical["drought"] == 1, xr.apply_ufunc(
    remove_short_droughts,
    historical["drought"],
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes = [np.float32],
    kwargs= {"min_duration": 30}), historical["drought"]
)

# remove short droughts from dis_extremes and volume_deficit
historical["dis_extremes"] = xr.where((historical["drought"] != 1) & (historical["dis_extremes"] != 2), np.nan, historical["dis_extremes"])
historical["volume_deficit"] = xr.where(
    (historical["drought"] != 1) & (historical["dis_extremes"] != 2), np.nan, historical["volume_deficit"]
)
historical["soilmoist_deficit"] = xr.where(
    (historical["drought"] != 1), np.nan, historical["soilmoist_deficit"]
)

historical["soil_drought"] = xr.where(
    (historical["drought"] != 1), np.nan, historical["soil_drought"]
)


#rechunk every variable appropriately
historical["dis_extremes"] = historical["dis_extremes"].chunk({"time":-1, "lat": 200, "lon": 200})
historical["volume_deficit"] = historical["volume_deficit"].chunk({"time":-1, "lat": 200, "lon": 200})

# Save to NetCDF with Compression
encoding = {var: {"zlib": True, "complevel": 4} for var in historical.data_vars}

with ProgressBar():
    historical.to_netcdf('Data/historical_flood_drought_dis>100.nc', encoding = encoding)

