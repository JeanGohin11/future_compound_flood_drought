"""This file detects drought and flood events in the future period/scenarios based on the historical thresholds, and stores the events
in a new file"""


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
    
    # Convert back to xarray DataArray
    return xr.DataArray(filtered_drought_np, dims=drought_array.dims, coords=drought_array.coords)

file1 = 'Data/historical_flood_drought_dis>100.nc'
file2 = xr.open_dataset('Data/ssp126_model_median_dis.nc')
file3 = xr.open_dataset('Data/ssp370_model_median_dis.nc')
file4 = xr.open_dataset('Data/ssp126_model_median_soilmoist.nc')
file5 = xr.open_dataset('Data/ssp370_model_median_soilmoist.nc')

#include ssp585
file6 = xr.open_dataset('Data/ssp585_model_median_dis.nc')
file7 = xr.open_dataset('Data/ssp585_model_median_soilmoist.nc')

chunk_scheme = {"time": -1, "lon": 200, "lat": 200}

ds_historical = xr.open_dataset(file1)
ds_future_ssp126 = xr.merge([file2, file4])
ds_future_ssp370 = xr.merge([file3, file5])

#ssp585
ds_future_ssp585 = xr.merge([file6, file7])

ds_historical = ds_historical.chunk(chunk_scheme)
ds_future_ssp126 = ds_future_ssp126.chunk(chunk_scheme)
ds_future_ssp370 = ds_future_ssp370.chunk(chunk_scheme)
ds_future_ssp585 = ds_future_ssp585.chunk(chunk_scheme)

ds_future = {"ssp126": ds_future_ssp126, "ssp370": ds_future_ssp370, "ssp585": ds_future_ssp585}


for scenario, ds in ds_future.items():

    # filter out grid cells with discharge > 100m3/s
    mask_high_discharge = (ds['dis'].groupby("time.year").mean(dim="time", skipna=True).mean(dim="year") > 100)

    for variable in ds.data_vars:
        ds[variable] = xr.where(mask_high_discharge, ds[variable], np.nan)

    #detect soil moisture droughts based on historical thresholds
    ds['soil_drought'] = xr.where(ds['soilmoist'] <= ds_historical['thresh_soilmoist'].sel(month=ds["time"].dt.month), np.int32(1), np.nan)
    ds["soil_drought"] = ds["soil_drought"].chunk(chunk_scheme)
    
    # compute soil moisture deficits
    ds["soilmoist_deficit"] = (ds["soilmoist"] - ds_historical['thresh_soilmoist'].sel(month=ds["time"].dt.month)).where(ds['soil_drought'] == 1).astype(np.float32)
    ds= ds.drop_vars('soilmoist')
    

    ds["soilmoist_deficit"] = ds["soilmoist_deficit"].chunk(chunk_scheme) #rechunk soilmoist_deficit


    #compute flood events across all cells based on historical thresholds
    ds['dis_extremes'] = xr.where(ds['dis'] >= ds_historical['flood_threshold'], np.float32(2), np.nan)
    ds["volume_deficit"] = xr.where(ds['dis_extremes']==2, ds['dis'] - ds_historical['flood_threshold'], np.nan)


    # compute hydrological drought events based on historical thresholds
    ds["dis_extremes"] = xr.where((ds['dis'] <= ds_historical['monthly_thresholds'].sel(month=ds["time"].dt.month)), np.float32(1),  ds["dis_extremes"])
    ds["dis_extremes"] = ds["dis_extremes"].chunk(chunk_scheme)

    # compute volume deficit for hydrological droughts
    ds["volume_deficit"] = xr.where(ds["dis_extremes"] == 1, (ds["dis"] - ds_historical["monthly_thresholds"].sel(month=ds["time"].dt.month)).astype(np.float32), ds["volume_deficit"])
    ds["volume_deficit"] = ds["volume_deficit"].chunk(chunk_scheme)
    
    #merge soil drought and hydrological drought
    ds["drought"] = xr.where(
    (ds["dis_extremes"] == 1) | (ds["soil_drought"] == 1), np.float32(1), np.nan)

    ds["drought"] = ds["drought"].chunk({"time": -1, "lat": 200, "lon": 200}) #rechunk drought variable

    #remove droughts shorter than 30 days
    ds["drought"] = xr.where(ds["drought"] == 1, xr.apply_ufunc(
        remove_short_droughts,
        ds["drought"],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes = [np.float32],
        kwargs= {"min_duration": 30}), ds["drought"]
    )

    # remove short droughts from dis_extremes and volume_deficit
    ds["dis_extremes"] = xr.where((ds["drought"] != 1) & (ds["dis_extremes"] != 2), np.nan, ds["dis_extremes"])
    ds["volume_deficit"] = xr.where(
    (ds["drought"] != 1) & (ds["dis_extremes"] != 2), np.nan, ds["volume_deficit"]
)
    ds["soilmoist_deficit"] = xr.where(
    (ds["drought"] != 1), np.nan, ds["soilmoist_deficit"]
)

    ds["soil_drought"] = xr.where(
    (ds["drought"] != 1), np.nan, ds["soil_drought"]
)

    
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    with ProgressBar():
         print("computing...")
        #  ds = ds.compute()
         ds.to_netcdf(f"Data/{scenario}_drought_flood_dis>100.nc", encoding = encoding)



