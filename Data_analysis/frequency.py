"""This file records the number of each CFD event type in each grid cell across the historical and future scenarios"""

import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar  

#prepare files
files = ["Data/historical_flood_drought_dis>100.nc", "Data/ssp126_drought_flood_dis>100.nc", "Data/ssp370_drought_flood_dis>100.nc", "Data/ssp585_drought_flood_dis>100.nc"]
scenarios = {"historical": files[0], "ssp126": files[1], "ssp370": files[2], "ssp585": files[3]}

for scenario, file in scenarios.items():
    # Load the dataset
    array = xr.open_dataset(file)
    chunk_scheme = {"lon": 200, "lat": 200}
    # Apply chunking scheme
    array = array.chunk({"time": -1, "lon": 200, "lat": 200})

    drought_end = (array["drought"] == 1) & (array['drought'].shift(time=-1) != 1)
    drought_start = (array["drought"] == 1) & (array['drought'].shift(time=1) != 1)
    # Identify drought and flood event starts/ends (masked)
    HD_end = (array["dis_extremes"] == 1) & (array["dis_extremes"].shift(time=-1) != 1)
    SMD_end = (array["soil_drought"] == 1) & (array["soil_drought"].shift(time=-1) != 1)

    HD_start = (array["dis_extremes"] == 1) & (array["dis_extremes"].shift(time=1) != 1)
    SMD_start = (array["soil_drought"] == 1) & (array["soil_drought"].shift(time=1) != 1)

    # Identify simultaneous drought & flood events (masked)
    FandD_events = (array["soil_drought"] == 1) & (array["dis_extremes"] == 2)
    FandD_event_starts = FandD_events.astype(int).diff(dim="time") == 1

    array["n_F&D"] = FandD_event_starts.sum(dim="time")
    array["n_F&D"] = array["n_F&D"].chunk(chunk_scheme)

    flood_start = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=1) != 2)
    flood_end   = (array["dis_extremes"] == 2) & (array["dis_extremes"].shift(time=-1) != 2)
    
    # Convert flood events to integer masks so that rolling sums can count them
    flood_start_int = flood_start.fillna(0).astype(int)
    flood_end_int   = flood_end.fillna(0).astype(int)
    
    # For the multiple events approach:
    # Count how many flood_end events occur in the 182 days after a drought ends.
    # We use rolling sum along the time dimension. The shift aligns the rolling window
    # such that at a drought_end time the count reflects the next 182 days.
    floods_after = flood_end_int[::-1].rolling(time=182, min_periods=1).sum().shift(time=1)[::-1]
    
    # Similarly, count how many flood_start events occur in the 182 days prior to a drought start.
    floods_before = flood_start_int.rolling(time=182, min_periods=1).sum().shift(time=1)
    
    # Count the number of drought-to-flood events:
    # At each drought_end, multiply by the number of flood_end events following within 6 months.
    array["n_DtoF_allD"] = (drought_end.fillna(0) * floods_after).sum(dim="time", skipna=True)
    
    # Count the number of flood-to-drought events:
    # At each drought_start, multiply by the number of flood_start events occurring in the previous 6 months.
    array["n_FtoD_allD"] = (drought_start.fillna(0) * floods_before).sum(dim="time", skipna=True)
    array["n_DtoF_HDonly"] = (HD_end.fillna(0) * floods_after).sum(dim="time", skipna=True)
    array["n_FtoD_HDonly"] = (HD_start.fillna(0) * floods_before).sum(dim="time", skipna=True)
    array["n_DtoF_SMDonly"] = (SMD_end.fillna(0) * floods_after).sum(dim="time", skipna=True)
    array["n_FtoD_SMDonly"] = (SMD_start.fillna(0) * floods_before).sum(dim="time", skipna=True)

    # Drop intermediate variables
    array = array.drop_vars(['dis', 'dis_extremes', 'soil_drought', 'volume_deficit', 'soilmoist_deficit', 'drought'])
    if scenario == 'historical':
        array = array.drop_vars(['thresh_soilmoist', 'flood_threshold', 'monthly_thresholds'])

    # Save output
    with ProgressBar():
        print(f"saving {scenario}")
        array.to_netcdf(f"Data/frequency_{scenario}_multiple_events.nc")
