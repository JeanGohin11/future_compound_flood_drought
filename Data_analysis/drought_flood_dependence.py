""" This file fits marginal distributions and copulas to drought and flood events in each grid cell (for each CFD type), and
selects the best one based on AIC. This is one for the historical period and future scenarios"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton, Gumbel, Frank
from dask.diagnostics import ProgressBar
import warnings

warnings.filterwarnings("ignore")

# Define distributions and copulas
MARGINAL_DISTS_FLOODS = {
    1: stats.genextreme,
    2: stats.lognorm,
    3: stats.genpareto,
    4: stats.pearson3
}

MARGINAL_DISTS_DROUGHTS = {
    1: stats.weibull_min,
    2: stats.gamma,
    3: stats.fisk,
    4: stats.lognorm,
    5: stats.expon
}
directions = ['DtoF', 'FtoD', 'D&F']
TRANSITION_TYPES = {
    'abrupt': (1, 30),
    'rapid': (31, 90),
    'moderate': (91, 182),
    'all': (1, 182)
}

COPULAS_bivariate = {1: GaussianMultivariate, 2: Clayton, 3: Gumbel, 4: Frank}


def compute_event_properties(data, merged_starts, merged_ends):
    """Compute properties of drought and flood events and assigns them to the entire run period in which the event is occuring."""
    # Create output arrays filled with NaNs
    cum_values = np.full_like(data, np.nan)
    max_values = np.full_like(data, np.nan)
    duration = np.full_like(data, np.nan)

    # Find all merged event boundaries
    starts = np.where(merged_starts)[0]
    ends = np.where(merged_ends)[0]
    assert len(starts) == len(ends), "Mismatched event boundaries!"
    assert all(s <= e for s, e in zip(starts, ends)), "Invalid event ordering!"

    for start_idx, end_idx in zip(starts, ends):
        # Extract data for this merged event period
        event_data = data[start_idx:end_idx + 1]

        # Calculate properties only on valid data
        valid_data = event_data[np.isfinite(event_data)]
        if valid_data.size == 0:
            continue

        cum_sum = np.nansum(valid_data)
        max_val = np.nanmax(valid_data)
        event_duration = end_idx - start_idx + 1
        # Note: We use start_idx:end_idx+1 to include the end index
        cum_values[start_idx:end_idx + 1] = cum_sum
        max_values[start_idx:end_idx + 1] = max_val
        duration[start_idx:end_idx + 1] = event_duration

    # Fill NaNs for periods without events
    return cum_values, max_values, duration

def fit_marginal_flood(data):
    """Fit marginal distributions and return best AIC fit"""
    best_aic = np.inf
    best_dist = np.nan
    best_params = None

    for name, dist in MARGINAL_DISTS_FLOODS.items():
        try:
            
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            if np.isfinite(log_likelihood):
                # Calculate AIC
                aic = 2 * len(params) - 2 * log_likelihood
                if aic < best_aic:
                    best_aic = aic
                    best_dist = name
                    best_params = params
        except Exception:
            continue

    return best_dist, best_params


def fit_marginal_drought(data):
    """Fit marginal distributions and return best AIC fit"""
    best_aic = np.inf
    best_dist = 0
    best_params = None
    data = np.abs(data)
    for dist_id, dist in MARGINAL_DISTS_DROUGHTS.items():
        try:
            params = dist.fit(np.abs(data))
            log_likelihood = np.sum(dist.logpdf(data, *params))
            if np.isfinite(log_likelihood):
                aic = 2 * len(params) - 2 * log_likelihood
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist_id
                    best_params = params
        except Exception:
            continue
    return best_dist, best_params


def compute_uniform(data, dist_name, params, floods=True):
    """Transform data to uniform margins using fitted distribution"""
    data = np.abs(data)
    dist = MARGINAL_DISTS_FLOODS[dist_name] if floods else MARGINAL_DISTS_DROUGHTS[dist_name]
    uniform = dist.cdf(data, *params)
    return np.clip(uniform, 1e-5, 1 - 1e-5)


def compute_kendall_tau(copula):
    """Compute Kendall's Tau for different copula types"""
    tau = np.nan
    if isinstance(copula, GaussianMultivariate):
        rho = copula.to_dict()['correlation']
        if isinstance(rho, list):
            rho = np.array(rho)
        rho = rho[0, 1]
        tau = (2 / np.pi) * np.arcsin(rho)
    else:
        tau = copula.tau
    return tau


def get_transition_pairs(drought_dates, flood_dates, sim_starts, drought_sev, flood_sev, event_type=None, simultaneous=False):
    """Identify drought-flood pairs with transition timing"""
    pairs = {}

    if simultaneous:
        pairs['D&F'] = {'drought': [], 'flood': []}
        sim_start_indices = np.where(sim_starts)[0]
        for idx in sim_start_indices:
            d_val = np.abs(drought_sev[idx])
            f_val = flood_sev[idx]
            if not np.isnan(f_val) and not np.isnan(d_val):
                pairs["D&F"]['drought'].append(d_val)
                pairs["D&F"]['flood'].append(f_val)
    else:
        for ttype in TRANSITION_TYPES:
            key = f"{ttype}_{event_type}"
            pairs[key] = {'drought': [], 'flood': []}

        for d_date in drought_dates:
            for f_date in flood_dates:
                if event_type == 'DtoF':
                    if f_date <= d_date:  # Require flood after drought
                        continue
                    delta = np.abs(f_date - d_date)
                else:
                    if d_date <= f_date:  # Require drought after flood
                        continue
                    delta = np.abs(d_date - f_date)
                d_val = np.abs(drought_sev[d_date])
                f_val = flood_sev[f_date]
                if np.isnan(f_val) or np.isnan(d_val):
                    continue
                for ttype, (min_d, max_d) in TRANSITION_TYPES.items():
                    if min_d <= delta <= max_d:
                        pairs[f"{ttype}_{event_type}"]['drought'].append(d_val)
                        pairs[f"{ttype}_{event_type}"]['flood'].append(f_val)
    return pairs

    

def grid_cell_analysis_all(
    drought_start, flood_start, drought_end, flood_end, sim_starts,
    drought_sev, flood_sev,
    fit_on_pairs=True
):
    ds_idx = np.where(drought_start)[0]
    de_idx = np.where(drought_end)[0]
    fs_idx = np.where(flood_start)[0]

    if fit_on_pairs:
        # pull together everything that eventually gets paired
        all_pair_d, all_pair_f = [], []
        # D→F
        p1 = get_transition_pairs(de_idx, fs_idx, sim_starts,
                                  drought_sev, flood_sev,
                                  event_type='DtoF')
        # F→D
        p2 = get_transition_pairs(ds_idx, np.where(flood_end)[0], sim_starts,
                                  drought_sev, flood_sev,
                                  event_type='FtoD')
        # Simul.
        p3 = get_transition_pairs(ds_idx, fs_idx, sim_starts,
                                  drought_sev, flood_sev,
                                  simultaneous=True)
        for pairs in (p1, p2):
            for key in pairs:
                all_pair_d += pairs[key]['drought']
                all_pair_f += pairs[key]['flood']
        # simul:
        all_pair_d += p3['D&F']['drought']
        all_pair_f += p3['D&F']['flood']

        drought_sample = np.array(all_pair_d)
        flood_sample   = np.array(all_pair_f)
    else:
        # fit on all events
        drought_sample = np.abs(drought_sev[ds_idx])
        flood_sample   = flood_sev[fs_idx]

    drought_sample = drought_sample[~np.isnan(drought_sample)]
    flood_sample   = flood_sample[~np.isnan(flood_sample)]

    #Fit marginals 
    if len(drought_sample) > 15:
        dist_d, params_d = fit_marginal_drought(drought_sample)
    else:
        dist_d, params_d = 0, None

    if len(flood_sample) > 15:
        dist_f, params_f = fit_marginal_flood(flood_sample)
    else:
        dist_f, params_f = 0, None

    copulas = np.full((len(directions), len(TRANSITION_TYPES)), 0, dtype=int)
    taus    = np.full((len(directions), len(TRANSITION_TYPES)), np.nan, dtype=float)

    if dist_d and dist_f:
        for i, dirn in enumerate(directions):
            if   dirn == 'DtoF':
                pairs = get_transition_pairs(de_idx, fs_idx, sim_starts,
                                             drought_sev, flood_sev,
                                             event_type='DtoF')
            elif dirn == 'FtoD':
                pairs = get_transition_pairs(ds_idx, np.where(flood_end)[0], sim_starts,
                                             drought_sev, flood_sev,
                                             event_type='FtoD')
            else:  # simultaneous
                pairs = get_transition_pairs(ds_idx, fs_idx, sim_starts,
                                             drought_sev, flood_sev,
                                             simultaneous=True)

            for j, ttype in enumerate(TRANSITION_TYPES):
                key = 'D&F' if dirn=='D&F' else f"{ttype}_{dirn}"
                d_vals = np.array(pairs[key]['drought'])
                f_vals = np.array(pairs[key]['flood'])
                if len(d_vals)>15 and len(f_vals)>15:
                    u_d = compute_uniform(d_vals, dist_d, params_d, floods=False)
                    u_f = compute_uniform(f_vals, dist_f, params_f, floods=True)

                    best_id, best_cop, best_aic = 0, None, np.inf
                    for cid, Cop in COPULAS_bivariate.items():
                        try:
                            cop = Cop()
                            cop.fit(np.vstack((u_d, u_f)).T)
                            ll  = np.sum(np.log(cop.pdf(np.vstack((u_d, u_f)).T)+1e-12))
                            aic = 2 - 2*ll
                            if aic<best_aic:
                                best_id, best_cop, best_aic = cid, cop, aic
                        except:
                            pass

                    copulas[i,j] = best_id
                    taus[i,j]    = compute_kendall_tau(best_cop) if best_cop else np.nan

    return dist_d, dist_f, copulas, taus



def apply_transitions(ds):
    drought_end = (ds['drought'] == 1) & (ds['drought'].shift(time=-1) != 1)
    drought_start = (ds['drought'] == 1) & (ds['drought'].shift(time=1) != 1)
    flood_end = (ds['dis_extremes'] == 2) & (ds['dis_extremes'].shift(time=-1) != 2)
    flood_start = (ds['dis_extremes'] == 2) & (ds['dis_extremes'].shift(time=1) != 2)

    simultaneous_events = (ds['dis_extremes'] == 2) & (ds['drought'] == 1)

    # get start of simultaneous events
    prev = simultaneous_events.shift(time=1, fill_value=False)
    sim_start = simultaneous_events & (~prev)

    hydro_deficit = xr.where(ds['volume_deficit'] <= 0, ds['volume_deficit'], np.nan)
    flood_volume = xr.where(ds['volume_deficit'] >= 0, ds['volume_deficit'], np.nan)
    _, _, drought_duration = xr.apply_ufunc(
        compute_event_properties,
        ds["soilmoist_deficit"],
        drought_start,
        drought_end,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time'], ['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64, np.float64, np.float64]
    )

    # Compute flood properties
    _, flood_magnitude, _ = xr.apply_ufunc(
        compute_event_properties,
        flood_volume,
        flood_start,
        flood_end,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time'], ['time'], ['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64, np.float64, np.float64]
    )

    results = xr.apply_ufunc(
            grid_cell_analysis_all,
            drought_start, flood_start, drought_end, flood_end, sim_start,
            drought_duration,  # Soil drought duration
            flood_magnitude,
            input_core_dims=[['time']] * 7,
            output_core_dims=[[], [], ['direction', 'transition'], ['direction', 'transition']] ,  # Now 8 outputs
            vectorize=True,
            dask='parallelized',
            output_sizes={'direction': len(directions), 'transition': len(TRANSITION_TYPES)},
            kwargs={'fit_on_pairs': True},
            output_dtypes=[int, int, int, float]
        )

    coords = {
        'lat': ds.lat,
        'lon': ds.lon,
        'direction': ('direction', directions),
        'transition': ('transition', list(TRANSITION_TYPES.keys()))}

    # Unpack results
    d_dist, f_dist, copulas, taus = results

    return xr.Dataset({
        'drought_dist': (('lat', 'lon'), d_dist.data),
        'flood_dist': (('lat', 'lon'), f_dist.data),
        'copula_type': (('lat', 'lon', 'direction', 'transition'), copulas.data),
        'kendalls_tau': (('lat', 'lon', 'direction', 'transition'), taus.data)
    }, coords=coords)


files = ["Data/historical_flood_drought_dis>100.nc",
         "Data/ssp126_drought_flood_dis>100.nc",
         "Data/ssp370_drought_flood_dis>100.nc"]

scenarios_data = {"historical": files[0], "ssp126": files[1], "ssp370": files[2]}
for scenario, file in scenarios_data.items():
    ds = xr.open_dataset(file)
    ds = ds.chunk({'time': -1, 'lat': 100, 'lon': 100})
    ds = apply_transitions(ds)

    # Save the results
    with ProgressBar():
        print(f"processing {scenario}")
        output_file = f"Data/{scenario}_drought_flood_dependence4.nc"
        ds.to_netcdf(output_file)