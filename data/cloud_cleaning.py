"""
Modified from https://github.com/seleneledain/forest_drought_forecasting/blob/main/data_downloading/cloud_cleaning.py
"""

import math
import warnings
from datetime import datetime
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter


def sanity_check(cube):
    for variable_name in cube.data_vars:
        if variable_name.startswith("s2_B"):
            data = cube[variable_name].values
            data[data < 0] = np.nan
            data[data > 1] = np.nan
            cube[variable_name] = (("time", "lat", "lon"), data)
        if variable_name.startswith("s2_ndvi"):
            data = cube[variable_name].values
            data[data < -1] = np.nan
            data[data > 1] = np.nan
            cube[variable_name] = (("time", "lat", "lon"), data)
    return cube


def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _, nC, nR = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nC * nR * ind + np.arange(nC * nR).reshape((nC, nR))
    return np.take(arr, idx)


def nan_percentile(inp, q):
    # from: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    arr = inp.copy()
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))

    # desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * (q / 100.0)
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    fc_equal_k_mask = f_arr == c_arr

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
    ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalue_from_index(
        arr=arr, ind=k_arr.astype(np.int32)
    )[
        fc_equal_k_mask
    ]  # if floor == ceiling take floor value

    return quant_arr


def process_group(x, s2_vars, pct):
    for s2_var in s2_vars:
        this_data = x[s2_var].values
        quant = nan_percentile(this_data, pct)
        above_quant = np.where(
            this_data >= np.expand_dims(quant, axis=0), this_data, np.nan
        )
        x[s2_var] = (("time", "lat", "lon"), above_quant)
    return x


def remove_lower_pct_per_week(cube, pct):
    """
    For each pixel timeseries and each Sentinel-2 variable, find pct percentile per week of year and remove values below.
    """
    s2_raw_ndvi = cube["s2_ndvi"].copy(deep=True)

    s2_vars = [name for name in list(cube.variables) if name.startswith("s2")]
    filtered_cloud = cube[s2_vars].where(
        (cube.s2_mask == 0) & cube.s2_SCL.isin([1, 2, 4, 5, 6, 7])
    )

    # Find lower X% per week of year for each pixel
    filtered_cloud["time"] = pd.to_datetime(filtered_cloud.time.values)
    filtered_ds = filtered_cloud.groupby(filtered_cloud.time.dt.isocalendar().week).map(
        partial(process_group, s2_vars=s2_vars, pct=pct)
    )
    cube[s2_vars] = filtered_ds

    cube["s2_raw_ndvi"] = s2_raw_ndvi
    cube["s2_cloud_cleaned_ndvi"] = cube["s2_ndvi"].copy(deep=True)

    return cube


def check_if_interp(data, max_nan):
    """
    For each pixel timeseries of NDVI, will check if there are sufficient values to allow timeseries smoothing (not too many NaN).

    :param data: xarray dataset with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    """

    # for variable_name in filtered_ds_10.data_vars:
    # if variable_name.startswith('s2'):

    # Select the variable (now just NDVI)
    variable = data["s2_ndvi"]
    ndvi = variable.values
    to_sample_init = np.zeros((ndvi.shape[1], ndvi.shape[2]), dtype=bool)
    to_sample = find_sample_mask(ndvi, max_nan, to_sample_init)

    # Initialise a new variable to store the data check
    new_variable = xr.DataArray(
        to_sample,
        dims=("lat", "lon"),
        coords={"lat": variable.lat, "lon": variable.lon},
    )

    # Add the new variable to the dataset
    data["to_sample"] = new_variable
    return data


@nb.jit(nopython=True)
def find_sample_mask(ndvi, max_nan, to_sample):
    for i in range(ndvi.shape[1]):
        for j in range(ndvi.shape[2]):
            arr = ndvi[:, i, j]
            if np.isnan(arr).all():
                continue
            # find maximum consecutive NaNs in a mask
            max_ = 0
            current = 0
            idx = 0
            while idx < arr.size:
                while idx < arr.size and math.isnan(arr[idx]):
                    current += 1
                    idx += 1
                if current > max_:
                    max_ = current
                current = 0
                idx += 1
            if max_ <= max_nan:
                to_sample[i, j] = True
    return to_sample


@nb.jit(nopython=True, cache=True, nogil=True)
def lowess(y, x, f=2.0 / 3.0, n_iter=3):
    """Lowess smoother (robust locally weighted regression).
    Fits a nonparametric regression curve to a scatterplot.
    From https://gist.github.com/ericpre/7a4dfba660bc8bb7499e7d96b8bdd4bb

    Parameters
    ----------
    y, x : np.ndarrays
        The arrays x and y contain an equal number of elements;
        each pair (x[i], y[i]) defines a data point in the
        scatterplot.
    f : float
        The smoothing span. A larger value will result in a
        smoother curve.
    n_iter : int
        The number of robustifying iteration. Thefunction will
        run faster with a smaller number of iterations.
    Returns
    -------
    yest : np.ndarray
        The estimated (smooth) values of y.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    w = np.minimum(
        1.0, np.maximum(np.abs((x.reshape((-1, 1)) - x.reshape((1, -1))) / h), 0.0)
    )
    w = (1 - w**3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for _ in range(n_iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)],
                ]
            )

            beta = np.linalg.lstsq(A, b)[0]
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta**2) ** 2

    return yest


@nb.jit(nopython=True, parallel=True, nogil=True)
def interpolate_variable(
    variable,
    to_sample,
    smoother="lowess",
    lowess_frac=0.07,
    SG_window_length=15,
    SG_polyorder=2,
):
    smoothed_values = np.copy(variable)
    time_numeric = np.arange(smoothed_values.shape[0])
    # Loop along the pixel dimensions (lat, lon)
    for i in nb.prange(smoothed_values.shape[1]):
        for j in range(smoothed_values.shape[2]):
            # Check if should smooth the timeseries for that pixel
            if to_sample[i, j]:
                # Replace with smoothed values
                y = smoothed_values[:, i, j]
                mask = np.isnan(y)
                # Perform smoothing with NaN handling
                if np.any(mask):  # Interpolate missing values
                    x = time_numeric[~mask]
                    y_interp = np.interp(
                        time_numeric, x, y[~mask]
                    )  # linear interpolation
                    sm_y = np.copy(y)
                    if smoother == "lowess":
                        sm_y = lowess(y_interp, time_numeric, f=lowess_frac)
                    # x = time_numeric[~mask]
                    # start, end = np.min(x), np.max(x)
                    # y_interp = np.interp(
                    #     time_numeric[start : end + 1], x, y[~mask]
                    # )  # linear interpolation
                    # sm_y = np.copy(y)
                    # if smoother == "lowess":
                    #     sm_y[start : end + 1] = lowess(
                    #         y_interp,
                    #         time_numeric[start : end + 1],
                    #         f=lowess_frac,
                    #     )
                    # elif smoother == "SG":
                    #     sm_y[start : end + 1] = savgol_filter(
                    #         y_interp,
                    #         window_length=SG_window_length,
                    #         polyorder=SG_polyorder,
                    #     )
                else:  # No missing values, perform smoothing directly
                    if smoother == "lowess":
                        sm_y = lowess(y, time_numeric, f=lowess_frac)
                    # elif smoother == "SG":
                    #     sm_y = savgol_filter(
                    #         y, window_length=SG_window_length, polyorder=SG_polyorder
                    #     )

                # Extract the smoothed values
                smoothed_values[:, i, j] = sm_y

    return smoothed_values


def smooth_s2_timeseries(
    cube,
    max_nan,
    remove_pct,
    smoother,
    lowess_frac,
    SG_window_length,
    SG_polyorder,
    pad_length=12,
):
    """
    Apply interpolation and smoothing to S2 timseries

    :param cube: xarray with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    :param pct: percentile (scaled between 0 and 1) under which to drop values per week of year
    :param frac: the fraction of the data used when estimating each y-value in the LOESS smoothing
    """
    cube = sanity_check(cube)

    # Filter out clouds and lower 5 percentile
    cube = remove_lower_pct_per_week(cube, remove_pct)
    print(f"Cloud cleaning - removed lower {remove_pct}% per week of yr")

    # First check if to interpolate or not
    cube = check_if_interp(cube, max_nan)
    print("Cloud cleaning - checked if interp")

    # Loop through variables and interpolate/smooth pixel timeseries for Sentinel-2
    for variable_name in cube.data_vars:
        if variable_name.startswith("s2_B") or variable_name.startswith(
            "s2_ndvi"
        ):  # only smooth continuous variables
            # Select the variable
            variable = cube[variable_name]

            # create padding months
            values = variable.values
            pad_before = np.zeros((pad_length, values.shape[1], values.shape[2]))
            pad_after = np.zeros((pad_length, values.shape[1], values.shape[2]))
            doy = cube.time.dt.dayofyear.values
            doy[doy == 366] = 365
            start_year_idx = np.array([i for i in range(len(doy) - 1) if doy[i] <= 5])
            end_year_idx = np.array([i for i in range(len(doy) - 1) if doy[i] >= 361])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(pad_length):
                    pad_after[i] = np.nanmedian(values[start_year_idx + i], axis=0)
                    pad_before[-i - 1] = np.nanmedian(values[end_year_idx - i], axis=0)
            padded_values = np.concatenate((pad_before, values, pad_after), axis=0)
            interp_values = interpolate_variable(
                padded_values,
                cube.to_sample.values,
                smoother=smoother,
                lowess_frac=lowess_frac,
                SG_window_length=SG_window_length,
                SG_polyorder=SG_polyorder,
            )
            interp_values = interp_values[pad_length:-pad_length]
            cube[variable_name] = (("time", "lat", "lon"), interp_values)

    return cube
