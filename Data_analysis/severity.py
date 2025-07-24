""" This file is used to compute all the severity variables (drought duration, flood magnitude, Empirical Compound Severity Index)
of all CFD event types for the historical period and future scenarios"""


import xarray as xr
import numpy as np
from scipy.ndimage import label
from dask.diagnostics import ProgressBar  


def compute_event_properties(data, merged_starts, merged_ends):
    """Compute cumulative volumes/deficits of floods droughts and store at the indices of occurrences of floods and droughts"""
    # Create output arrays filled with NaNs
    cum_values = np.full_like(data, np.nan)
    max_values = np.full_like(data, np.nan)
    
    # Find all merged event boundaries
    starts = np.where(merged_starts)[0]
    ends = np.where(merged_ends)[0]
    assert len(starts) == len(ends), "Mismatched event boundaries!"
    assert all(s <= e for s, e in zip(starts, ends)), "Invalid event ordering!"

    for start_idx, end_idx in zip(starts, ends):
        # Extract data for this merged event period
        event_data = data[start_idx:end_idx+1]
        
        # Calculate properties only on valid data
        valid_data = event_data[np.isfinite(event_data)]
        if valid_data.size == 0:
            continue
        
        cum_sum = np.nansum(valid_data)
        max_val = np.nanmax(valid_data)

        # Note: We use start_idx:end_idx+1 to include the end index
        cum_values[start_idx:end_idx+1] = cum_sum
        max_values[start_idx:end_idx+1] = max_val
    
    return cum_values, max_values

def compute_sev_variable(events1, events2, sev_variable, **kwargs):
    """Compute severity variable based on event proximity and type."""
    # Initialize empty lists to store severities
    severities_all = []
    severities_slow = []
    severities_rap = []
    severities_abr = []

    if kwargs.get("DtoF", False):
        # Get indices of drought and flood events
        drought_indices = np.where(events1)[0]
        flood_indices = np.where(events2)[0]
        # Loop through each drought event
        for d in drought_indices: 
            # All floods within 182 days *after* the drought
            valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 182)]
            if valid_floods.size > 0:
                severity = sev_variable[valid_floods]
                severities_all.extend(severity)
                valid_floods = flood_indices[(flood_indices > d + 90) & (flood_indices <= d + 182)]
                if valid_floods.size > 0:
                    severity = sev_variable[valid_floods]
                    severities_slow.extend(severity)
                valid_floods = flood_indices[(flood_indices > d + 30) & (flood_indices <= d + 90)]
                if valid_floods.size > 0:
                    severity = sev_variable[valid_floods]
                    severities_rap.extend(severity)
                valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 30)]
                if valid_floods.size > 0:
                    severity = sev_variable[valid_floods]
                    severities_abr.extend(severity) 

    if kwargs.get("FtoD", False):
        drought_indices = np.where(events1)[0]
        flood_indices = np.where(events2)[0]
        for d in drought_indices:

                # All floods within 182 days *before* the drought
                valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 182)]
                if valid_floods.size > 0:
                    severity = sev_variable[valid_floods]
                    severities_all.extend(severity)
                
                    valid_floods = flood_indices[(flood_indices < d - 90) & (flood_indices >= d - 182)]
                    if valid_floods.size > 0:
                        severity = sev_variable[valid_floods]
                        severities_slow.extend(severity)
                    
                    valid_floods = flood_indices[(flood_indices < d - 30) & (flood_indices >= d - 90)]
                    if valid_floods.size > 0:
                        severity = sev_variable[valid_floods]
                        severities_rap.extend(severity)

                # All floods within 30 days before the drought
                    valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 30)]
                    if valid_floods.size > 0:
                        severity = sev_variable[valid_floods]
                        severities_abr.extend(severity)

    if kwargs.get("D&F", False):
        severities = []
        # Combine all relevant event points into one boolean mask

        event_mask = events1 | events2
        valid_indices = np.where(event_mask & simultaneous_events)[0]
        severities.extend(sev_variable[valid_indices].tolist())

    MAX_events = 1500
    severity_all = np.array(severities_all, dtype=np.float64)
    severity_slow = np.array(severities_slow, dtype=np.float64)
    severity_rap = np.array(severities_rap, dtype=np.float64)
    severity_abr = np.array(severities_abr, dtype=np.float64)
    for severity in [severity_all, severity_slow, severity_rap, severity_abr]:
    # Ensure the output is padded to MAX_events
        output = np.full(MAX_events, np.nan) 
        severities = []
        if severity.size > 0:
            n = min(MAX_events, len(severity))
            output[:n] = severity[:n]
            severities.append(output)
    return severities[0], severities[1], severities[2], severities[3]


def compute_exceedance_probabilities(historical_severity, current_severity):
    """Compute non-exceedance probabilities based on historical data."""

    current_severity = np.abs(current_severity)
    historical_severity = np.abs(historical_severity)

    # Remove NaN values from current severity
    valid_mask = ~np.isnan(current_severity)
    valid_severities = current_severity[valid_mask]

    # Handle empty historical data case
    historical_non_nan = historical_severity[~np.isnan(historical_severity)]
    if len(historical_non_nan) == 0:
        return np.full_like(current_severity, np.nan)  # Return all NaNs

    sorted_severities = np.sort(historical_non_nan)
    ecdf_values = np.arange(1, len(sorted_severities) + 1) / len(sorted_severities)

    # Handle empty current data case
    if valid_severities.size == 0:
        return np.full_like(current_severity, np.nan)

    unique_severities = np.unique(valid_severities)
    unique_indices = np.searchsorted(sorted_severities, unique_severities, side='right')
    
    # Ensure valid indices for empty edge case
    max_idx = len(ecdf_values) - 1 if len(ecdf_values) > 0 else 0
    unique_indices = np.clip(unique_indices, 0, max_idx)
    
    unique_probs = ecdf_values[unique_indices]

    severity_to_prob = dict(zip(unique_severities, unique_probs))
    non_exceedance_probs = np.full_like(current_severity, np.nan)
    non_exceedance_probs[valid_mask] = [severity_to_prob.get(val, np.nan) for val in valid_severities]

    return non_exceedance_probs



def compute_ECSI(simultaneous_events, events1, events2, non_exceedance_probs_floods, non_exceedance_probs_hydrodroughts,
                 non_exceedance_probs_soildroughts, time_values, **kwargs):
    """Compute ECSI values based on event proximity and type."""
    # convert nan values to zero for computation
    non_exceedance_probs_floods = np.nan_to_num(non_exceedance_probs_floods)
    non_exceedance_probs_hydrodroughts = np.nan_to_num(non_exceedance_probs_hydrodroughts)
    non_exceedance_probs_soildroughts = np.nan_to_num(non_exceedance_probs_soildroughts)
    ECSI = []
    transition_times = []
    ECWA = []

    if kwargs.get("DtoF", False):
        drought_indices = np.where(events1)[0]
        flood_indices = np.where(events2)[0]

        td_list = []
        ecsi_list = []
        ecwa_list = []
        for d in drought_indices:
            # All floods within 182 days *after* the drought
            valid_floods = flood_indices[(flood_indices > d) & (flood_indices <= d + 182)]
            if valid_floods.size == 0:
                continue

            # Compute time differences (flood happens after drought)
            tds = (time_values[valid_floods] - time_values[d]).astype("timedelta64[D]").astype(np.float32)

            # Vectorized severity calculation
            drought_severity = np.sqrt((non_exceedance_probs_hydrodroughts[d]**2 + non_exceedance_probs_soildroughts[d]**2)/2)
            flood_severities = non_exceedance_probs_floods[valid_floods]

            severities = np.sqrt((drought_severity ** 2 + flood_severities ** 2) / 2)
            ecwa = np.arctan(flood_severities/drought_severity) - np.pi/4

            # Append in bulk
            td_list.extend(tds.tolist())
            ecsi_list.extend(severities.tolist())
            ecwa_list.extend(ecwa.tolist())

        transition_times.extend(td_list)
        ECSI.extend(ecsi_list)
        ECWA.extend(ecwa_list)

    if kwargs.get("FtoD", False):
        drought_indices = np.where(events1)[0]
        flood_indices = np.where(events2)[0]

        # Prepare empty lists to collect results
        td_list = []
        ecsi_list = []
        ecwa_list = []

        for d in drought_indices:
            # Get all floods within the 182-day window before the drought
            valid_floods = flood_indices[(flood_indices < d) & (flood_indices >= d - 182)]
            if valid_floods.size == 0:
                continue

            # Vectorized time difference computation
            tds = (time_values[d] - time_values[valid_floods]).astype("timedelta64[D]").astype(np.float32)

            # Vectorized severity computation
            drought_severity = np.sqrt((non_exceedance_probs_hydrodroughts[d]**2 + non_exceedance_probs_soildroughts[d]**2)/2)
            flood_severities = non_exceedance_probs_floods[valid_floods]
            severities = np.sqrt((drought_severity ** 2 + flood_severities ** 2) / 2)
            ecwa = np.arctan(flood_severities/drought_severity) - np.pi/4

            # Append all results at once
            td_list.extend(tds.tolist())
            ecsi_list.extend(severities.tolist())
            ecwa_list.extend(ecwa.tolist())

        transition_times.extend(td_list)
        ECSI.extend(ecsi_list)
        ECWA.extend(ecwa_list)

    if kwargs.get("D&F", False):

        event_mask = events1 | events2
        valid_indices = np.where(event_mask & simultaneous_events)[0]
        transition_times.extend([0.0] * len(valid_indices))
            # Compute time difference

        drought_severity = np.sqrt((non_exceedance_probs_hydrodroughts[valid_indices]**2 + non_exceedance_probs_soildroughts[valid_indices]**2)/2)
        # Compute severity only at valid simultaneous event indices
        flood_severity = non_exceedance_probs_floods[valid_indices]

        severity_vals = np.sqrt((       drought_severity**2
                                        + flood_severity**2
                                ) / 2)
        ecwa_vals = np.arctan(flood_severity/drought_severity) - np.pi/4

        ECSI.extend(severity_vals.tolist())
        ECWA.extend(ecwa_vals.tolist())
    
    MAX_TRANSITIONS = 1500
    times_padded = np.full(MAX_TRANSITIONS, np.nan, dtype=np.float32)
    ecsi_padded = np.full(MAX_TRANSITIONS, np.nan, dtype=np.float32)
    ecwa_padded = np.full(MAX_TRANSITIONS, np.nan, dtype=np.float32)

    ECSI_values = np.array(ECSI, dtype=np.float64)
    transition_times = np.array(transition_times, dtype=np.float32)
    ecwa_values = np.array(ECWA, dtype=np.float64)

    n = min(len(transition_times), MAX_TRANSITIONS)
    if n > 0:
        times_padded[:n] = transition_times[:n]
        ecsi_padded[:n] = ECSI_values[:n]
        ecwa_padded[:n] = ecwa_values[:n]

    if kwargs.get("DtoF", False) or kwargs.get("FtoD", False):
        return times_padded, ecsi_padded, ecwa_padded
    else:
        return ecsi_padded, ecwa_padded
    

def compute_drought_duration(drought_array, simultaneous_events, flood_events, **kwargs):
    """
    Compute drought durations based on proximity to flood events and simultaneous flood-droughts.
    
    Parameters:
    - drought_array: 1D array with 1s for drought, NaNs or 0 elsewhere
    - simultaneous_events: 1D boolean array
    - drought_start / drought_end: 1D boolean arrays with event flags
    - flood_starts: 1D boolean array with flood starts
    - kwargs: control logic for DtoF, FtoD, DandF flags

    Returns:
    - List of valid drought durations
    """

    flood_indices = np.where(flood_events)[0]

    labeled_array, n_events = label(drought_array == 1)

    # Initialize result containers
    dtof_all, dtof_mod, dtof_rap, dtof_abr = [], [], [], []
    ftod_all, ftod_mod, ftod_rap, ftod_abr = [], [], [], []
    dandf = []

    for label_id in range(1, n_events + 1):
        indices = np.where(labeled_array == label_id)[0]
        start_idx = indices[0]
        end_idx = indices[-1]
        duration = end_idx - start_idx + 1

        if kwargs.get("DtoF", False):
            after = flood_indices[(flood_indices > end_idx) & (flood_indices <= end_idx + 182)]
            if after.size > 0:
                dtof_all.append(duration)
            if np.any((after > end_idx + 90) & (after <= end_idx + 182)):
                dtof_mod.append(duration)
            if np.any((after > end_idx + 30) & (after <= end_idx + 90)):
                dtof_rap.append(duration)
            if np.any((after > end_idx) & (after <= end_idx + 30)):
                dtof_abr.append(duration)

        if kwargs.get("FtoD", False):
            before = flood_indices[(flood_indices < start_idx) & (flood_indices >= start_idx - 182)]
            if before.size > 0:
                ftod_all.append(duration)
            if np.any((before < start_idx - 90) & (before >= start_idx - 182)):
                ftod_mod.append(duration)
            if np.any((before < start_idx - 30) & (before >= start_idx - 90)):
                ftod_rap.append(duration)
            if np.any((before >= start_idx - 30) & (before < start_idx)):
                ftod_abr.append(duration)

        if kwargs.get("D&F", False):
            if np.any(simultaneous_events[start_idx:end_idx + 1]):
                dandf.append(duration)

    # Convert to padded arrays
    
    def pad(arr):
        out = np.full(1500, np.nan)
        out[:len(arr)] = arr[:1500]
        return np.array(out, dtype=np.float64)

    
    if kwargs.get("DtoF", False):
        return pad(dtof_all), pad(dtof_mod), pad(dtof_rap), pad(dtof_abr)
    if kwargs.get("FtoD", False):
        return pad(ftod_all), pad(ftod_mod), pad(ftod_rap), pad(ftod_abr)
    if kwargs.get("D&F", False):
        return pad(dandf)



# Define the scenarios and their corresponding files
files = ["Data/historical_flood_drought_dis>100.nc", 
         "Data/ssp126_drought_flood_dis>100.nc", 
         "Data/ssp370_drought_flood_dis>100.nc",
         "Data/ssp585_drought_flood_dis>100.nc"]
scenarios = {"historical": files[0], "ssp126": files[1], "ssp370": files[2], "ssp585": files[3]}




##################################################################################################################### HISTORICAL
# Load the dataset
array = xr.open_dataset(files[0])
array = array.chunk({"time": -1, "lat": 100, "lon": 100})

soil_deficit = xr.where(array["soilmoist_deficit"] < 0, array["soilmoist_deficit"], np.nan)
hydro_deficit = xr.where(array['volume_deficit'] < 0, array['volume_deficit'], np.nan)
flood_volume = xr.where(array['volume_deficit'] > 0, array['volume_deficit'], np.nan)


# Identify event start/end points
drought_end = (array['drought'] == 1) & (array['drought'].shift(time=-1) != 1)
drought_start = (array['drought'] == 1) & (array['drought'].shift(time=1) != 1)

flood_end = (array['dis_extremes'] == 2) & (array['dis_extremes'].shift(time=-1) != 2)
flood_start = (array['dis_extremes'] == 2) & (array['dis_extremes'].shift(time=1) != 2)
simultaneous_events = (array['dis_extremes'] == 2) & (array['soil_drought'] == 1)

#get start of simultaneous events

prev = simultaneous_events.shift(time=1, fill_value=False)
sim_start = simultaneous_events & (~prev)

time_values = array["time"].values 


# Compute drought properties across all cells
soil_drought_cum_deficit_historical, soil_drought_intensity_historical = xr.apply_ufunc(
    compute_event_properties,
    soil_deficit,
    drought_start,
    drought_end,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[['time'], ['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[np.float64, np.float64]
)

hydro_drought_cum_deficit_historical, hydro_drought_intensity_historical = xr.apply_ufunc(
    compute_event_properties,
    hydro_deficit,
    drought_start,
    drought_end,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[['time'], ['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[np.float64, np.float64]
)


# Compute flood properties
flood_cum_volume_historical, flood_magnitude_historical = xr.apply_ufunc(
    compute_event_properties,
    flood_volume,
    flood_start,
    flood_end,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[['time'], ['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[np.float64, np.float64]
)


#compute severities of individual droughts and floods

array["severities_hydrodroughts_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, 
                                                        drought_end, flood_start, hydro_drought_cum_deficit_historical, 
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        dask="parallelized",
                                                        vectorize=True,
                                                        kwargs= {"DtoF":True, "drought":True},
                                                        output_dtypes=[np.float64])

array["severities_hydrodroughts_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, hydro_drought_cum_deficit_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"FtoD":True, "drought":True},
                                                        output_dtypes=[np.float64])
array["severities_SMdroughts_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, soil_drought_cum_deficit_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"D&F":True},
                                                        output_dtypes=[np.float64])
array["severities_SMdroughts_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, soil_drought_cum_deficit_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"DtoF":True, "drought":True},
                                                        output_dtypes=[np.float64])
array["severities_SMdroughts_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, soil_drought_cum_deficit_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"FtoD":True, "drought":True},
                                                        output_dtypes=[np.float64])

array["severities_floods_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, flood_cum_volume_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"D&F":True},
                                                        output_dtypes=[np.float64])
array["severities_floods_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, flood_cum_volume_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"DtoF":True, "flood":True},
                                                        output_dtypes=[np.float64])
array["severities_floods_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, flood_cum_volume_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"FtoD":True, "flood":True},
                                                        output_dtypes=[np.float64])

# Compute flood magnitudes
array["magnitude_flood_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, flood_magnitude_historical,
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"DtoF":True, "flood":True},
                                                        output_dtypes=[np.float64])

array["magnitude_flood_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, flood_magnitude_historical, 
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"FtoD":True, "flood":True},
                                                        output_dtypes=[np.float64])

array["magnitude_flood_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, flood_magnitude_historical,    
                                                        input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_sev"]],
                                                        output_sizes={"events_sev": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"D&F":True},
                                                        output_dtypes=[np.float64])
                            
# Compute drought durations                            
array["duration_DtoF"], array["duration_DtoF_mod"], array["duration_DtoF_rap"], array["duration_DtoF_abr"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_start,  
                                                        input_core_dims=[["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_duration"], ["events_duration"], ["events_duration"], ["events_duration"]],
                                                        output_sizes={"events_duration": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"DtoF":True},
                                                        output_dtypes=[np.float64, np.float64, np.float64, np.float64])

array["duration_FtoD"], array["duration_FtoD_mod"], array["duration_FtoD_rap"], array["duration_FtoD_abr"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_end,
                                                        input_core_dims=[["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_duration"], ["events_duration"], ["events_duration"], ["events_duration"]],
                                                        output_sizes={"events_duration": 1500},
                                                        vectorize=True,
                                                        dask="parallelized",
                                                        kwargs= {"FtoD":True},
                                                        output_dtypes=[np.float64, np.float64, np.float64, np.float64])

array["duration_D&F"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_start,
                                                        input_core_dims=[["time"], ["time"], ["time"]],
                                                        output_core_dims=[["events_duration"]],
                                                        output_sizes={"events_duration": 1500},
                                                        vectorize = True,
                                                        dask="parallelized",
                                                        kwargs= {"D&F":True},
                                                        output_dtypes=[np.float64])


# Compute non-exceedance probabilities

non_exceed_probs_sev_hydrodroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                hydro_drought_cum_deficit_historical, hydro_drought_cum_deficit_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])

non_exceed_probs_sev_SMDdroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                soil_drought_cum_deficit_historical, soil_drought_cum_deficit_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])

non_exceed_probs_sev_floods = xr.apply_ufunc(compute_exceedance_probabilities,
                                                flood_cum_volume_historical, flood_cum_volume_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])


non_exceed_probs_intens_hydrodroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                hydro_drought_intensity_historical, hydro_drought_intensity_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])

non_exceed_probs_intens_SMDdroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                soil_drought_intensity_historical, soil_drought_intensity_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])

non_exceed_probs_magn_floods = xr.apply_ufunc(compute_exceedance_probabilities,
                                                flood_magnitude_historical, flood_magnitude_historical,
                                                input_core_dims=[["time"], ["time"]],
                                                output_core_dims=[["time"]],
                                                vectorize=True,
                                                dask="parallelized",
                                                output_dtypes=[np.float64])


#Compute ECSI values


array["transition_times_DtoF"], array["ECSI_sev_DtoF"], array["ECWA_sev_DtoF"] = xr.apply_ufunc(compute_ECSI, sim_start, 
                                    drought_end, flood_start, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                    input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                    output_core_dims=[["events_sev"], ["events_sev"], ["events_sev"]],
                                    output_sizes={"events_sev": 1500},
                                    vectorize=True,
                                    dask="parallelized",
                                    kwargs={"DtoF":True},
                                    output_dtypes=[np.float64, np.float64, np.float64])

array["transition_times_FtoD"], array["ECSI_sev_FtoD"], array["ECWA_sev_FtoD"] = xr.apply_ufunc(compute_ECSI,
                                sim_start, 
                                    drought_start, flood_end, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                    input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                    output_core_dims=[["events_sev"], ["events_sev"], ["events_sev"]],
                                    output_sizes={"events_sev": 1500},
                                    vectorize = True, 
                                    dask="parallelized",
                                    kwargs={"FtoD":True},
                                    output_dtypes=[np.float64, np.float64, np.float64])

array["ECSI_sev_D&F"], array["ECWA_sev_D&F"] = xr.apply_ufunc(compute_ECSI,
                                sim_start, 
                                    drought_start, flood_start, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                    input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                    output_core_dims=[["events_sev"], ["events_sev"]],
                                    output_sizes={"events_sev": 1500},
                                    vectorize = True, 
                                    dask="parallelized",
                                    kwargs={"D&F":True},
                                    output_dtypes=[np.float64, np.float64])

array = array.drop_vars(['dis', 'dis_extremes', 'soil_drought', 'volume_deficit', 'soilmoist_deficit', 'drought', 'thresh_soilmoist', 'flood_threshold', 'monthly_thresholds'])

# After computing historical properties, rename 'time' dimension
# to 'time_historical' so that it can be used to compute the exceedance probabilities
# of the future datasets based on the historical distribution
hydro_drought_cum_deficit_historical = hydro_drought_cum_deficit_historical.rename({'time': 'time_historical'})
soil_drought_cum_deficit_historical = soil_drought_cum_deficit_historical.rename({'time': 'time_historical'})
flood_cum_volume_historical = flood_cum_volume_historical.rename({'time': 'time_historical'})
hydro_drought_intensity_historical = hydro_drought_intensity_historical.rename({'time': 'time_historical'})
soil_drought_intensity_historical = soil_drought_intensity_historical.rename({'time': 'time_historical'})
flood_magnitude_historical = flood_magnitude_historical.rename({'time': 'time_historical'})

#convert array to netcdf
# Ensure all variables have consistent dimensions and set encodings
encoding = {var: {"zlib": True, "complevel": 4} for var in array.data_vars}


with ProgressBar():
    array.to_netcdf("Data/historical_severity.nc", encoding=encoding)


########################################################################################################FUTURE SCENARIOS
# Load the dataset
# Loop through each scenario and compute the required metrics
# Load the historical dataset for exceedance probabilities
for scenario, file in scenarios.items():
    if scenario == 'historical':
        continue
    # Load the dataset  
    print(f"Processing {scenario}...")
    array = xr.open_dataset(file)
    array = array.chunk({"time": -1, "lat": 100, "lon": 100})

    soil_deficit = xr.where(array["soilmoist_deficit"] <= 0, array["soilmoist_deficit"], np.nan)
    hydro_deficit = xr.where(array['volume_deficit'] <= 0, array['volume_deficit'], np.nan)
    flood_volume = xr.where(array['volume_deficit'] >= 0, array['volume_deficit'], np.nan)


    # Identify event start/end points
    drought_end = (array['drought'] == 1) & (array['drought'].shift(time=-1) != 1)
    drought_start = (array['drought'] == 1) & (array['drought'].shift(time=1) != 1)
    
    flood_end = (array['dis_extremes'] == 2) & (array['dis_extremes'].shift(time=-1) != 2)
    flood_start = (array['dis_extremes'] == 2) & (array['dis_extremes'].shift(time=1) != 2)
    simultaneous_events = ((array['dis_extremes'] == 2) & (array['soil_drought'] == 1))

    #get start of simultaneous events
    prev = simultaneous_events.shift(time=1, fill_value=False)
    sim_start = simultaneous_events & (~prev)

    time_values = array["time"].values
    # Compute drought properties
    soil_drought_cum_deficit, soil_drought_intensity = xr.apply_ufunc(
        compute_event_properties,
        soil_deficit,
        drought_start,
        drought_end,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize = True, 
        dask='parallelized',
        output_dtypes=[np.float64, np.float64]
    )

    hydro_drought_cum_deficit, hydro_drought_intensity = xr.apply_ufunc(
        compute_event_properties,
        hydro_deficit,
        drought_start,
        drought_end,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize = True, 
        dask='parallelized',
        output_dtypes=[np.float64, np.float64]
    )


    # Compute flood properties
    flood_cum_volume, flood_magnitude = xr.apply_ufunc(
        compute_event_properties,
        flood_volume,
        flood_start,
        flood_end,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize = True, 
        dask='parallelized',
        output_dtypes=[np.float64, np.float64]
    )

    # compute individual drought/flood properties for each CFD type
    array["severities_hydrodroughts_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, 
                                                            drought_end, flood_start, hydro_drought_cum_deficit, 
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"DtoF":True, "drought":True},
                                                            output_dtypes=[np.float64])

    array["severities_hydrodroughts_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, hydro_drought_cum_deficit,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"FtoD":True, "drought":True},
                                                            output_dtypes=[np.float64])
    
    array["severities_SMdroughts_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, soil_drought_cum_deficit,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"D&F":True},                                                           
                                                            output_dtypes=[np.float64])
    
    array["severities_SMdroughts_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, soil_drought_cum_deficit,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"DtoF":True, "drought":True},
                                                            output_dtypes=[np.float64])
    
    array["severities_SMdroughts_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, soil_drought_cum_deficit,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"FtoD":True, "drought":True},
                                                            output_dtypes=[np.float64])

    array["severities_floods_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, flood_cum_volume,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"D&F":True},
                                                            output_dtypes=[np.float64])
    
    array["severities_floods_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, flood_cum_volume,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"DtoF":True, "flood":True},
                                                            output_dtypes=[np.float64])
    
    array["severities_floods_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, flood_cum_volume,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"FtoD":True, "flood":True},
                                                            output_dtypes=[np.float64])

    # Compute flood magnitude
    array["magnitude_flood_DtoF"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_end, flood_start, flood_magnitude,
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"DtoF":True, "flood":True},
                                                            output_dtypes=[np.float64])
    
    array["magnitude_flood_FtoD"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_end, flood_magnitude, 
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"FtoD":True, "flood":True},
                                                            output_dtypes=[np.float64])

    array["magnitude_flood_D&F"] = xr.apply_ufunc(compute_sev_variable, sim_start, drought_start, flood_start, flood_magnitude,    
                                                            input_core_dims=[["time"], ["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_sev"]],
                                                            output_sizes={"events_sev": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"D&F":True},
                                                            output_dtypes=[np.float64])
                                                            
    #compute drought duration
    array["duration_DtoF"], array["duration_DtoF_mod"], array["duration_DtoF_rap"], array["duration_DtoF_abr"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_start,  
                                                            input_core_dims=[["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_duration"], ["events_duration"], ["events_duration"], ["events_duration"]],
                                                            output_sizes={"events_duration": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"DtoF":True},
                                                            output_dtypes=[np.float64, np.float64, np.float64, np.float64])
    
    array["duration_FtoD"], array["duration_FtoD_mod"], array["duration_FtoD_rap"], array["duration_FtoD_abr"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_end,
                                                            input_core_dims=[["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_duration"], ["events_duration"], ["events_duration"], ["events_duration"]],
                                                            output_sizes={"events_duration": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"FtoD":True},
                                                            output_dtypes=[np.float64, np.float64, np.float64, np.float64])
    
    array["duration_D&F"] = xr.apply_ufunc(compute_drought_duration, array["drought"], sim_start, flood_start,
                                                            input_core_dims=[["time"], ["time"], ["time"]],
                                                            output_core_dims=[["events_duration"]],
                                                            output_sizes={"events_duration": 1500},
                                                            vectorize = True, 
                                                            dask="parallelized",
                                                            kwargs= {"D&F":True},
                                                            output_dtypes=[np.float64])

    #Compute exceedance probabilities
    non_exceed_probs_sev_hydrodroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    hydro_drought_cum_deficit_historical, hydro_drought_cum_deficit,
                                                    input_core_dims=[['time_historical'], ["time"]],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])

    non_exceed_probs_sev_SMDdroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    soil_drought_cum_deficit_historical, soil_drought_cum_deficit,
                                                    input_core_dims=[['time_historical'], ["time"]],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])

    non_exceed_probs_sev_floods = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    flood_cum_volume_historical, flood_cum_volume,
                                                    input_core_dims=[['time_historical'], ["time"]],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])


    non_exceed_probs_intens_hydrodroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    hydro_drought_intensity_historical, hydro_drought_intensity,
                                                    input_core_dims=[['time_historical'], ['time']],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])

    non_exceed_probs_intens_SMDdroughts = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    soil_drought_intensity_historical, soil_drought_intensity,
                                                    input_core_dims=[['time_historical'], ['time']],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])

    non_exceed_probs_magn_floods = xr.apply_ufunc(compute_exceedance_probabilities,
                                                    flood_magnitude_historical, flood_magnitude,
                                                    input_core_dims=[['time_historical'], ["time"]],
                                                    output_core_dims=[["time"]],
                                                    vectorize = True, 
                                                    dask="parallelized",
                                                    output_dtypes=[np.float64])


    # Compute ECSI

    array["transition_times_DtoF"], array["ECSI_sev_DtoF"], array["ECWA_sev_DtoF"]  = xr.apply_ufunc(compute_ECSI, sim_start, 
                                        drought_end, flood_start, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                        output_core_dims=[["events_sev"], ["events_sev"], ["events_sev"]],
                                        output_sizes={"events_sev": 1500},
                                        vectorize = True, 
                                        dask="parallelized",
                                        kwargs={"DtoF":True},
                                        output_dtypes=[np.float64, np.float64, np.float64])

    array["transition_times_FtoD"], array["ECSI_sev_FtoD"], array["ECWA_sev_FtoD"]= xr.apply_ufunc(compute_ECSI,
                                    sim_start, 
                                        drought_start, flood_end, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                        output_core_dims=[["events_sev"], ["events_sev"], ["events_sev"]],
                                        output_sizes={"events_sev": 1500},
                                        vectorize = True, 
                                        dask="parallelized",
                                        kwargs={"FtoD":True},
                                        output_dtypes=[np.float64, np.float64, np.float64])

    array["ECSI_sev_D&F"], array["ECWA_sev_D&F"] = xr.apply_ufunc(compute_ECSI,
                                    sim_start, 
                                        drought_start, flood_start, non_exceed_probs_sev_floods, non_exceed_probs_sev_hydrodroughts, non_exceed_probs_sev_SMDdroughts, time_values,
                                        input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
                                        output_core_dims=[["events_sev"], ["events_sev"]],
                                        output_sizes={"events_sev": 1500},
                                        vectorize = True, 
                                        dask="parallelized",
                                        kwargs={"D&F":True},
                                        output_dtypes=[np.float64, np.float64])
    
    # Drop unnecessary variables
    array = array.drop_vars(['dis', 'dis_extremes', 'soil_drought', 'volume_deficit', 'soilmoist_deficit', 'drought'])
    
    with ProgressBar():
        array.to_netcdf(f"Data/{scenario}_severity.nc", encoding = encoding)
    

