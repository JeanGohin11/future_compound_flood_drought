""" This file records the lag of all FtoD and DtoF events in each grid cell, for historical period and future scenarios"""

import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar  

files = ["Data/historical_flood_drought_dis>100.nc", 
         "Data/ssp126_drought_flood_dis>100.nc", 
         "Data/ssp370_drought_flood_dis>100.nc",
         "Data/ssp585_drought_flood_dis>100.nc"]


scenarios = {"historical": files[0], "ssp126": files[1], "ssp370": files[2], "ssp585": files[3]}

def compute_transition_times(drought_events, flood_events, time_values, DtoF=True):
    """
    Compute transition times (days) between drought and flood events within a 182-day window.
    
    Returns a fixed-size array of 1000 elements padded with NaT if necessary.
    """
    drought_indices = np.where(drought_events)[0]
    flood_indices = np.where(flood_events)[0]
    
    transition_times = []
    
    if DtoF:
        for d in drought_indices:
            valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 182)]
            if valid_floods.size > 0:
                td = (time_values[valid_floods] - time_values[d]).astype("timedelta64[D]").astype(np.float32)
                transition_times.extend(td)
    else:
        for d in drought_indices:
            valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 182)]
            if valid_floods.size > 0:
                for f in valid_floods:
                    td = (time_values[d] - time_values[valid_floods]).astype("timedelta64[D]").astype(np.float32)
                    transition_times.extend(td)
    
    transition_times = np.array(transition_times, dtype=np.float32)
    MAX_TRANSITIONS = 1500
    output = np.full(MAX_TRANSITIONS, np.nan)  # Use NaN for missing values
    
    if transition_times.size > 0:
        n = min(len(transition_times), MAX_TRANSITIONS)
        output[:n] = transition_times[:n]
    
    return output


for scenario, file in scenarios.items():
    array = xr.open_dataset(file)
    chunk_scheme = {"lon": 200, "lat": 200}
    array = array.chunk({"time": -1, "lon": 200, "lat": 200})
    
    array["drought"] = xr.where((array["dis_extremes"] == 1) | (array["soil_drought"] == 1), np.float32(1), np.nan)
    
    drought_end = (array["drought"] == 1) & (array["drought"].shift(time=-1) != 1)
    drought_start = (array["drought"] == 1) & (array["drought"].shift(time=1) != 1)
    flood_start = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=1) != 2)
    flood_end = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=-1) != 2)
    
    time_values = array["time"].values  # Ensure this is a numpy datetime64 array
    print("Time values dtype:", time_values.dtype)

    # Apply transition time computation
    array["transition_times_DtoF"] = xr.apply_ufunc(
        compute_transition_times, 
        drought_end, flood_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["transition_times"]],
        vectorize=True,
        dask="parallelized",
        output_sizes={"transition_times": 1500},
        kwargs={"DtoF": True},
        output_dtypes=[np.float32]
    )

    array["transition_times_FtoD"] = xr.apply_ufunc(
        compute_transition_times, 
        drought_start, flood_end, time_values,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["transition_times"]],
        vectorize=True,
        output_sizes={"transition_times": 1500},
        kwargs={"DtoF": False},
        dask="parallelized",
        output_dtypes=[np.float32]
    )

    array = array.drop_vars(['dis', 'dis_extremes', 'soil_drought', 'volume_deficit', 'soilmoist_deficit', 'month', 'drought'])
    if scenario == 'historical':
        array = array.drop_vars(['thresh_soilmoist', 'flood_threshold', 'monthly_thresholds'])
    
    array["rapid_trans_DtoF_count"] = ((array["transition_times_DtoF"] < 90) & (array["transition_times_DtoF"] > 30)).sum(dim="transition_times", skipna=True) 
    array["abrupt_trans_DtoF_count"] = (array["transition_times_DtoF"] <= 30).sum(dim="transition_times", skipna=True) 
    array["rapid_trans_FtoD_count"] = ((array["transition_times_FtoD"] < 90) & (array["transition_times_FtoD"] > 30)).sum(dim="transition_times", skipna=True) 
    array["abrupt_trans_FtoD_count"] = (array["transition_times_FtoD"] <= 30).sum(dim="transition_times", skipna=True) 
    
    
    with ProgressBar():
        print(f"Saving {scenario}")
        array.to_netcdf(f"Data/transition_times_{scenario}.nc")



    
