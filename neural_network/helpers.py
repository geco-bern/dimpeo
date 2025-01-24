import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pyproj import Transformer
import xarray as xr
import dask.array as da
import dask_image.ndfilters as dimg
import torch
import torch.nn as nn


START_YEAR = 2017
END_YEAR = 2023
NUM_DATAPOINTS_PER_YEAR = 73
H, W = 128, 128


def get_doy(dates):
    return np.array(
            [
                pd.to_datetime(d).to_pydatetime().timetuple().tm_yday
                for d in dates
            ],
            dtype=int,
    )


def get_split_indices(years):
    indices = [np.arange((y - START_YEAR) * NUM_DATAPOINTS_PER_YEAR, (y - START_YEAR - 1) * NUM_DATAPOINTS_PER_YEAR) for y in years]
    return np.concatenate(indices, axis=0)


def check_missing_timestamps(cube, max_conseq_dates=2):
    """Check for missing timestamps in cube.

    Args:
        cube (xr.Dataset): Cube to check for missing timestamps.
        max_conseq_dates (int): Maximum number of consecutive missing timestamps to allow.

    Returns:
        missing_dates (list): List of missing timestamps
    """
    timestamps = cube.time.values
    missing_dates = []

    # beginning of 2017
    current_timestamp = timestamps[0]
    while (current_timestamp - np.timedelta64(5, "D")).astype("datetime64[Y]").astype(
        int
    ) + 1970 >= START_YEAR:
        current_timestamp -= np.timedelta64(5, "D")
        missing_dates.append(current_timestamp)

    # end of 2023
    current_timestamp = timestamps[-1]
    while (current_timestamp + np.timedelta64(5, "D")).astype("datetime64[Y]").astype(
        int
    ) + 1970 <= END_YEAR:
        current_timestamp += np.timedelta64(5, "D")
        missing_dates.append(current_timestamp)

    current_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]
    nr_conseq_dates_max = 0
    while current_timestamp < last_timestamp:
        # Check for presence of next timestamp at 5 days interval
        expected_date = current_timestamp + np.timedelta64(5, "D")
        if expected_date not in timestamps:
            missing_dates.append(expected_date)
            # Record number of consecutive missing timestamps
            if len(missing_dates) > 1 and (
                missing_dates[-1] - missing_dates[-2]
            ) == np.timedelta64(5, "D"):
                nr_conseq_dates_max += 1
            else:
                nr_conseq_dates_max = 1
        current_timestamp = expected_date

    if nr_conseq_dates_max > max_conseq_dates:
        print(f"Warning: Too many consecutive missing dates ({nr_conseq_dates_max})")

    return missing_dates


def create_reference_raster(filepath, channel_name, channel_coords, res=20, bounds=(2484000, 1075000, 2834000, 1296000)):
    """
    Create a reference raster with given dimensions and transformation.
    """
    # bounds are outside CH extreme points: left, bottom, right, top
    # source: https://de.wikipedia.org/wiki/Geographische_Extrempunkte_der_Schweiz
    width = (bounds[2] - bounds[0]) // res
    height = (bounds[3] - bounds[1]) // res

    N_coords = np.linspace(bounds[3] - res // 2, bounds[1] + res // 2, height)
    E_coords = np.linspace(bounds[0] + res // 2, bounds[2] - res // 2, width)

    reference_tmp = da.zeros((len(channel_coords), height, width), dtype="float32", chunks=(20, 2000, 2000))
    count_tmp = da.zeros((len(channel_coords), height, width), dtype="uint16", chunks=(20, 2000, 2000))
    forest_mask_tmp = da.zeros((height, width), dtype="bool", chunks=(2000, 2000))
    
    ds = xr.Dataset(
        {
            "reference_tmp": ((channel_name, "N", "E"), reference_tmp),
            "count_tmp": ((channel_name, "N", "E"), count_tmp),
            "forest_mask": (("N", "E"), forest_mask_tmp)
        },
        coords={
            channel_name: channel_coords,
            "N": N_coords,
            "E": E_coords,
        },
    )

    ds.attrs["crs"] = "EPSG:2056"
    ds.attrs["negative_anomaly_id"] = 0
    ds.attrs["normal_id"] = 1
    ds.attrs["positive_anomaly_id"] = 2
    ds.attrs["missing_id"] = 255

    ds.to_zarr(filepath, compute=True)
    return ds


def project_patch(filepath, lon_left, lon_right, lat_bottom, lat_top, patch, mask, group_zarr, channel_name, nx=128, ny=128):
    """
    Project and resample patch onto the reference raster.
    """
    lon_grid = np.linspace(lon_left, lon_right, nx)
    lat_grid = np.linspace(lat_top, lat_bottom, ny)

    # Transform the lat/lon grid to SwissTopo coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x_st, y_st = transformer.transform(lon_grid, lat_grid)

    # Flatten the source grid and patch data for interpolation
    src_x, src_y = np.meshgrid(x_st, y_st)
    src_points = np.array([src_y.ravel(), src_x.ravel()]).T

    patch_values = patch.reshape(patch.shape[0], -1)
    mask_values = mask.reshape(-1)

    # Determine the bounding box for the subraster
    delta_x = (np.max(x_st) - np.min(x_st)) / (nx - 1)
    delta_y = (np.max(y_st) - np.min(y_st)) / (ny - 1)
    min_y_st, min_x_st = np.min(y_st) - delta_y / 2, np.min(x_st) - delta_x / 2
    max_y_st, max_x_st = np.max(y_st) + delta_y / 2, np.max(x_st) + delta_x / 2

    # Extract the subraster and corresponding coordinates
    min_x_idx = (group_zarr.coords["E"] >= min_x_st).argmax().item()
    max_x_idx = (group_zarr.coords["E"] > max_x_st).argmax().item() - 1
    min_y_idx = (group_zarr.coords["N"] < min_y_st).argmax().item() - 1
    max_y_idx = (group_zarr.coords["N"] <= max_y_st).argmax().item()
    
    subraster = group_zarr.isel(N=slice(max_y_idx, min_y_idx), E=slice(min_x_idx, max_x_idx))

    sub_x = subraster.coords["E"].values
    sub_y = subraster.coords["N"].values
    dst_x, dst_y = np.meshgrid(sub_x, sub_y)
    dst_points = np.array([dst_y.ravel(), dst_x.ravel()]).T

    subraster["forest_mask"] = (("N", "E"), griddata(src_points, mask_values, dst_points, method="nearest").reshape((subraster.dims["N"], subraster.dims["E"])))

    # Interpolate the patch data to fit the reference raster grid
    for i in range(patch.shape[0]):
        # griddata ignores NaN values
        valid_mask = griddata(src_points, ~np.isnan(patch_values[i, :]), dst_points, method="nearest").reshape((subraster.dims["N"], subraster.dims["E"]))
        resampled_patch = griddata(src_points, np.nan_to_num(patch_values[i, :]), dst_points, method="nearest").reshape((subraster.dims["N"], subraster.dims["E"]))
        subraster["reference_tmp"][i, ...] += resampled_patch
        subraster["count_tmp"][i, ...] += valid_mask
        
    subraster.to_zarr(filepath, region={"N": slice(max_y_idx, min_y_idx), "E": slice(min_x_idx, max_x_idx), channel_name: slice(0, subraster.dims[channel_name])})


def apply_gaussian_filter(dask_data, sigma):
    """
    Apply Gaussian filter to the data with handling NaNs.
    """
    nan_mask = da.isnan(dask_data)
    data_filled = da.nan_to_num(dask_data)
    smoothed_data = dimg.gaussian_filter(data_filled, sigma=sigma)
    smoothed_data = da.where(nan_mask, np.nan, smoothed_data)
    return smoothed_data


def group_by_month(data):
    num_chunks = 12
    chunk_size = data.shape[0] // num_chunks
    # Split the data into 12 chunks using slicing 
    data_chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks - 1)]
    data_chunks.append(data[(num_chunks - 1) * chunk_size:])
    means = [da.nanmean(chunk, axis=0) for chunk in data_chunks]
    return da.stack(means, axis=0).rechunk(chunks=[-1, 2000, 2000])


@np.errstate(invalid='ignore')
def consolidate(file_path, channel_name, channel_coords, post_processing=True, discretize=True):
    group_zarr = xr.open_zarr(file_path)
    raster, count = group_zarr["reference_tmp"], group_zarr["count_tmp"]
    data = da.where(count != 0, raster / count, np.nan)

    if not channel_name in group_zarr.coords:
        group_zarr = group_zarr.assign_coords({channel_name: channel_coords})

    if post_processing:
        # Define the sigmas for Gaussian smoothing
        # space_sigma = 50
        # time_sigma = 3

        # Apply Gaussian smoothing using Dask
        # smoothed_dask_data = apply_gaussian_filter(data, (time_sigma, space_sigma, space_sigma)).rechunk(20, 2000, 2000)

        # Instead, group by month
        smoothed_dask_data = group_by_month(data)

        # for anomalies:
        # 0 = negative anomaly
        # 1 = normal
        # 2 = positive anomaly
        # 255 = missing value
        if discretize:
            data = discretize_anomalies(smoothed_dask_data)
        else:
            data = smoothed_dask_data

    group_zarr["data"] = ((channel_name, "N", "E"), data)
    group_zarr.to_zarr(file_path, mode="a")
    clean_up(file_path)


def clean_up(file_path):
    ds = xr.open_zarr(file_path, drop_variables=["reference_tmp", "count_tmp", "tmp"])
    ds.to_zarr(file_path.replace(".zarr", "_tmp.zarr"))
    shutil.rmtree(file_path)
    shutil.move(file_path.replace(".zarr", "_tmp.zarr"), file_path)


def get_dates():
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def discretize_anomalies(data, threshold=1.5, missing_index=255):
    out = da.full(data.shape, fill_value=missing_index, dtype="uint8")
    out = da.where(data < -threshold, 0, out)
    out = da.where(data > threshold, 2, out)
    out = da.where((data >= -threshold) & (data <= threshold), 1, out)
    return out.rechunk(chunks=[-1, 2000, 2000])


def convert_params(params):
    sos = params[:, 0]
    eos = params[:, 2] + nn.functional.softplus(params[:, 3])
    sndvi = params[:, 4]
    wndvi = params[:, 5]
    return torch.stack([sos, eos, sndvi, wndvi], dim=1)
