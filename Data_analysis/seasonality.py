""" This file extracts the seasonality variables from the drought and flood events of each CFD types in each grid cell for
the historical period and future scenarios"""

import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

#define functions for seasonality

# Function to convert dates to radians
def convert_date_to_radians(drought_start, drought_end, flood_start, flood_end, time_values):
    """
    Convert start dates of drought and flood events to radians for seasonality analysis.
    
    Parameters:
    - drought_start: 1D boolean or integer array (1s at drought starts)
    - flood_start: same as above but for flood starts
    - time_values: xarray or pandas datetime array
    
    Returns:
    - drought_radians, flood_radians: arrays of angles in radians
    """
    radians_drought = np.full_like(drought_start, np.nan, dtype=float)  # Initialize with NaN
    radians_flood = np.full_like(flood_start, np.nan, dtype=float)  # Initialize with NaN
    # Extract dates where drought or flood starts
    
    droughtstart_indices = np.where(drought_start)[0]
    droughtend_indices = np.where(drought_end)[0]
    floodstart_indices = np.where(flood_start)[0]
    floodend_indices = np.where(flood_end)[0]
    time_values = pd.to_datetime(time_values)

    assert len(droughtstart_indices) == len(droughtend_indices), "Mismatched event boundaries!"
    assert all(s <= e for s, e in zip(droughtstart_indices, droughtend_indices)), "Invalid event ordering!"
    for i, j in zip(droughtstart_indices, droughtend_indices):
        date = time_values[i]
        doy = date.dayofyear  # Get day of year as int
        year_length = 366 if date.is_leap_year else 365
        theta = ((doy - 0.5) * 2 * np.pi) / year_length
        radians_drought[i:j+1] = theta

    assert len(floodstart_indices) == len(floodend_indices), "Mismatched event boundaries!"
    assert all(s <= e for s, e in zip(floodstart_indices, floodend_indices)), "Invalid event ordering!"
    for i, j in zip(floodstart_indices, floodend_indices):
        date = time_values[i]
        doy = date.dayofyear
        year_length = 366 if date.is_leap_year else 365
        theta = ((doy - 0.5) * 2 * np.pi) / year_length
        radians_flood[i:j+1] = theta


    return radians_drought, radians_flood

# Function to compute cosine and sine components from radians
def compute_cosine_sine_components(radians):
    """
    Compute cosine and sine components for a given set of angles in radians.
    
    Parameters:
    - radians: array of angles in radians
    
    Returns:
    - cos_component, sin_component: arrays of cosine and sine components
    """
    
    n_drought = np.count_nonzero(~np.isnan(radians[0]))
    n_flood = np.count_nonzero(~np.isnan(radians[1]))

    if n_drought == 0:
        cos_component_droughts = np.nan
        sin_component_droughts = np.nan
    else:
        cos_component_droughts = np.nansum(np.cos(radians[0])) / n_drought
        sin_component_droughts = np.nansum(np.sin(radians[0])) / n_drought

    if n_flood == 0:
        cos_component_floods = np.nan
        sin_component_floods = np.nan
    else:
        cos_component_floods = np.nansum(np.cos(radians[1]))/n_flood
        sin_component_floods = np.nansum(np.sin(radians[1]))/n_flood
    
    return cos_component_droughts, sin_component_droughts, cos_component_floods, sin_component_floods

# Function to compute mean date of occurrence from cosine and sine components
def compute_mean_date_occurence(cos_component, sin_component):
    """
    Compute mean date of occurrence from cosine and sine components.
    
    Parameters:
    - cos_component: cosine component
    - sin_component: sine component
    
    Returns:
    - mean_date: mean date of occurrence in radians
    """
    θ = np.arctan2(sin_component, cos_component)

    # shift negative angles into [0, 2π)
    if θ < 0:
        θ = θ + 2 * np.pi

    # now θ is in [0,2π), map to months 0–12
    mean_date = θ * 12 / (2 * np.pi)

    return mean_date

# Function to compute strength of seasonality from cosine and sine components
def compute_strength_seasonality(cos_component, sin_component):
    """
    Compute strength of seasonality from cosine and sine components.
    
    Parameters:
    - cos_component: cosine component
    - sin_component: sine component
    
    Returns:
    - strength_seasonality: strength of seasonality
    """
    strength_seasonality = np.sqrt(cos_component**2 + sin_component**2)
    return strength_seasonality

# Function to compute seasonality for drought and flood events in CFDs (combines all functions)
def compute_seasonality(drought_start, drought_end, flood_start, flood_end, simultaneous_events, time_values, **kwargs):
    drought_radians, flood_radians = convert_date_to_radians(drought_start, drought_end, flood_start, flood_end, time_values)
    

    drought_radians_inCFD = []
    flood_radians_inCFD = []
    if kwargs.get("DtoF", False):
        drought_indices = np.where(drought_end)[0]
        flood_indices = np.where(flood_start)[0]
        for d in drought_indices:
            if kwargs.get("all", False):
                valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 182)]
            elif kwargs.get("moderate", False):
                valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 182) & (flood_indices > d + 90)]
            elif kwargs.get("rapid", False):
                valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 90) & (flood_indices > d + 30)]
            elif kwargs.get("abrupt", False):
                valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 30)]
            if valid_floods.size > 0:
                drought_radians_inCFD.append(drought_radians[d])
                flood_radians_inCFD.extend(flood_radians[valid_floods])
    

    elif kwargs.get("FtoD", False):
        drought_indices = np.where(drought_start)[0]
        flood_indices = np.where(flood_end)[0]
        for d in drought_indices:
            if kwargs.get("all", False):
                valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 182)]
            elif kwargs.get("moderate", False):
                valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 182) & (flood_indices < d - 90)]
            elif kwargs.get("rapid", False):
                valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 90) & (flood_indices < d - 30)]
            elif kwargs.get("abrupt", False):
                valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 30)]
            if valid_floods.size > 0:
                flood_radians_inCFD.extend(flood_radians[valid_floods])
                drought_radians_inCFD.append(drought_radians[d])
    
    elif kwargs.get("D&F", False):
        event_mask = drought_start | flood_start
        valid_indices = np.where(event_mask & simultaneous_events)[0]
        drought_radians_inCFD.extend(drought_radians[valid_indices])
        flood_radians_inCFD.extend(flood_radians[valid_indices])

    cos_drought, sin_drought, cos_flood, sin_flood = compute_cosine_sine_components([drought_radians_inCFD, flood_radians_inCFD])

    mean_date_drought = compute_mean_date_occurence(cos_drought, sin_drought)
    mean_date_flood = compute_mean_date_occurence(cos_flood, sin_flood)
    strength_drought = compute_strength_seasonality(cos_drought, sin_drought)
    strength_flood = compute_strength_seasonality(cos_flood, sin_flood)

    return mean_date_drought, strength_drought, mean_date_flood, strength_flood
        

files = ["Data/historical_flood_drought_dis>100.nc", 
         "Data/ssp126_drought_flood_dis>100.nc", 
         "Data/ssp370_drought_flood_dis>100.nc",
         "Data/ssp585_drought_flood_dis>100.nc"] 

scenarios_data = {"historical": files[0], "ssp126": files[1], "ssp370": files[2], "ssp585": files[3]}

for scenario, data in scenarios_data.items():
    # Load the dataset
    array = xr.open_dataset(data)
    chunk_scheme = {"time": -1, "lon": 100, "lat": 100}
    # Apply chunking scheme
    array = array.chunk(chunk_scheme)

    # Identify drought and flood event starts/ends (masked)
    drought_start = (array["drought"] == 1) & (array['drought'].shift(time=1) != 1)
    flood_start = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=1) != 2)
    
    drought_end = (array["drought"] == 1) & (array['drought'].shift(time=-1) != 1)
    flood_end = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=-1) != 2)

    # Identify simultaneous drought & flood events (masked)
    simultaneous_events = (array["drought"] == 1) & (array["dis_extremes"] == 2)

    prev = simultaneous_events.shift(time=1, fill_value=False)
    sim_start = simultaneous_events & (~prev) #take start of simultaneous events

    time_values = array["time"].values 

    # compute seasonality for DtoF
    array["date_drought_DtoF"], array["strength_seas_drought_DtoF"], array["date_flood_DtoF"], array["strength_seas_flood_DtoF"],  = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        kwargs = {"DtoF": True, "all": True},
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
    )

    # compute seasonality for DtoF with different timeframes

    array["date_drought_DtoF_moderate"], array["strength_seas_drought_DtoF_moderate"], array["date_flood_DtoF_moderate"], array["strength_seas_flood_DtoF_moderate"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"DtoF": True, "moderate": True},
    )

    array["date_drought_DtoF_rapid"], array["strength_seas_drought_DtoF_rapid"], array["date_flood_DtoF_rapid"], array["strength_seas_flood_DtoF_rapid"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"DtoF": True, "rapid": True},
    )

    array["date_drought_DtoF_abrupt"], array["strength_seas_drought_DtoF_abrupt"], array["date_flood_DtoF_abrupt"], array["strength_seas_flood_DtoF_abrupt"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"DtoF": True, "abrupt": True},
    )


    # compute seasonality for FtoD
    array["date_drought_FtoD"], array["strength_seas_drought_FtoD"], array["date_flood_FtoD"], array["strength_seas_flood_FtoD"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"FtoD": True, "all": True},
    )

    # compute seasonality for FtoD with different timeframes
    array["date_drought_FtoD_moderate"], array["strength_seas_drought_FtoD_moderate"], array["date_flood_FtoD_moderate"], array["strength_seas_flood_FtoD_moderate"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"FtoD": True, "moderate": True},
    )

    array["date_drought_FtoD_rapid"], array["strength_seas_drought_FtoD_rapid"], array["date_flood_FtoD_rapid"], array["strength_seas_flood_FtoD_rapid"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"FtoD": True, "rapid": True},
    )

    array["date_drought_FtoD_abrupt"], array["strength_seas_drought_FtoD_abrupt"], array["date_flood_FtoD_abrupt"], array["strength_seas_flood_FtoD_abrupt"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"FtoD": True, "abrupt": True},
    )


    # compute seasonality for D&F
    array["date_drought_D&F"], array["strength_seas_drought_D&F"], array["date_flood_D&F"], array["strength_seas_flood_D&F"] = xr.apply_ufunc(
        compute_seasonality, drought_start, drought_end, flood_start, flood_end, sim_start, time_values,
        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64, np.float64],
        kwargs = {"D&F": True},
    )

    # Drop unnecessary variables
    array = array.drop_vars(['dis', 'dis_extremes', 'soil_drought', 'volume_deficit', 'soilmoist_deficit', 'drought'])
    if scenario =="historical":
        array = array.drop_vars(['thresh_soilmoist', 'flood_threshold', 'monthly_thresholds'])
    # Save the modified dataset to a new NetCDF file
    with ProgressBar():
        array.to_netcdf(f"Data/seasonality_{scenario}.nc")
    


