import shutil
import numpy as np
from scipy.interpolate import griddata
from pyproj import Transformer
import xarray as xr
import dask.array
import os

from utils.helpers import check_missing_timestamps


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

    reference_tmp = dask.array.zeros((len(channel_coords), height, width), dtype="float32", chunks=(20, 1800, 1800))
    count_tmp = dask.array.zeros((len(channel_coords), height, width), dtype="uint16", chunks=(20, 1800, 1800))
    forest_mask_tmp = dask.array.zeros((height, width), dtype="bool", chunks=(1800, 1800))
    
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
    

def project_patch(filepath, lon_left, lon_right, lat_bottom, lat_top, patch, mask, group_zarr, channel_name, subraster_margin=3, nx=128, ny=128):
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
    min_y_st, min_x_st = np.min(y_st), np.min(x_st)
    max_y_st, max_x_st = np.max(y_st), np.max(x_st)

    # Extract the subraster and corresponding coordinates
    min_x_idx = (group_zarr.coords["E"] > min_x_st).argmax().item() - 1 - subraster_margin
    max_x_idx = (group_zarr.coords["E"] > max_x_st).argmax().item() + 1 + subraster_margin
    min_y_idx = (group_zarr.coords["N"] < min_y_st).argmax().item() + 1 + subraster_margin
    max_y_idx = (group_zarr.coords["N"] < max_y_st).argmax().item() - 1 - subraster_margin
    
    subraster = group_zarr.isel(N=slice(max_y_idx, min_y_idx), E=slice(min_x_idx, max_x_idx))

    sub_x = subraster.coords["E"].values
    sub_y = subraster.coords["N"].values
    dst_x, dst_y = np.meshgrid(sub_x, sub_y)
    dst_points = np.array([dst_y.ravel(), dst_x.ravel()]).T

    subraster["forest_mask"] = (("N", "E"), griddata(src_points, mask_values, dst_points, method='nearest').reshape((subraster.dims["N"], subraster.dims["E"])))

    # Interpolate the patch data to fit the reference raster grid
    for i in range(patch.shape[0]):
        resampled_patch = griddata(src_points, patch_values[i, :], dst_points, method='nearest').reshape((subraster.dims["N"], subraster.dims["E"]))
        # Add resampled patch to reference raster
        subraster["reference_tmp"][i, ...] += np.nan_to_num(resampled_patch)
        subraster["count_tmp"][i, ...] += ~np.isnan(resampled_patch)
        
    subraster.to_zarr(filepath, region={"N": slice(max_y_idx, min_y_idx), "E": slice(min_x_idx, max_x_idx), channel_name: slice(0, subraster.dims[channel_name])})


def map_patches_to_raster(filepath, patches, coords, masks, group_zarr, channel_name):
    """
    Map patches onto the reference raster.
    """
    for patch, c, mask in zip(patches, coords, masks):
        project_patch(filepath, c[0], c[1], c[2], c[3], patch, mask, group_zarr, channel_name)


def consolidate(file_path, group_zarr):
    raster, count = group_zarr["reference_tmp"], group_zarr["count_tmp"]
    data = xr.where(count != 0, raster / count, np.nan)
    # for anomalies: 
    # 0 = negative anomaly
    # 1 = normal
    # 2 = positive anomaly
    # 255 = missing value
    group_zarr["data"] = discretize_anomalies(data)
    group_zarr.to_zarr(file_path, mode="a")
    clean_up(file_path)


def clean_up(file_path):
    ds = xr.open_zarr(file_path, drop_variables=["reference_tmp", "count_tmp"])
    ds.to_zarr(file_path.replace(".zarr", "_tmp.zarr"))
    shutil.rmtree(file_path)
    shutil.move(file_path.replace(".zarr", "_tmp.zarr"), file_path)


def get_dates(year):
    minicube = xr.open_dataset("/data_2/dimpeo/cubes/2017_1_10_2023_12_30_7.212724933477382_46.627329370567224_128_128_raw.nc", engine="h5netcdf")
    missing_dates = check_missing_timestamps(minicube, 2017, 2023)
    if missing_dates:
        minicube = minicube.reindex(
            time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
        )
    return minicube.time[(year - 2017) * 73:(year - 2016) * 73].values


def discretize_anomalies(data, missing_index=255):
    data = xr.where(data < 0, -1, data)
    data = xr.where(data > 0, 1, data)
    data = data.fillna(missing_index - 1)
    data += 1
    return data.astype("uint8")


if __name__ == "__main__":

    YEAR = 2023
    COORDS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/coords_nolon_era_500k.npy"
    PARAMS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/params_nolon_era_500k.npy"
    ANOMS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/anoms_nolon_era_500k.npy"
    MASK_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/forest_masks.npy"

    out_path = "/data_2/scratch/dbrueggemann/nn/overlay"

    # output of `extract_maps.py`
    # N is the number of cubes
    # coords contains: (lon_left, lon_right, lat_bottom, lat_top) for each cube
    coords = np.load(COORDS_FILENAME)  # N x 4
    forest_masks = np.load(MASK_FILENAME)
    
    # Parameter predictions first
    # print("Doing parameters")
    # params = np.load(PARAMS_FILENAME)  # N x 128 x 128 x 6
    # params_filename = os.path.join(out_path, "parameters.zarr")
    # group_zarr, transform = create_reference_raster(
    #     filepath=params_filename,
    #     channel_name="param_layer",
    #     channel_coords=["SOS", "EOS", "sNDVI", "wNDVI"])
    # map_patches_to_raster(params, coords, group_zarr, transform)
    # correct_overlap(group_zarr)
    # del params

    # Anomalies next
    print("Doing anomalies")
    anoms = np.load(ANOMS_FILENAME)  # N x 73 x 128 x 128
    dates = get_dates(YEAR)
    anoms_filename = os.path.join(out_path, "anomalies.zarr")
    channel_name = "time"
    group_zarr = create_reference_raster(filepath=anoms_filename, channel_name=channel_name, channel_coords=dates)
    map_patches_to_raster(anoms_filename, anoms, coords, forest_masks, group_zarr, channel_name=channel_name)
    consolidate(anoms_filename, group_zarr)
