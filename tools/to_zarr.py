import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from pyproj import Transformer
import zarr
import os


def create_reference_raster(filepath, channels, res=20, bounds=(2484000, 1075000, 2834000, 1296000)):
    """
    Create a reference raster with given dimensions and transformation.
    """
    # bounds are outside CH extreme points: left, bottom, right, top
    # source: https://de.wikipedia.org/wiki/Geographische_Extrempunkte_der_Schweiz
    width = (bounds[2] - bounds[0]) // res
    height = (bounds[3] - bounds[1]) // res

    transform = from_origin(bounds[0], bounds[3], res, res)

    store = zarr.DirectoryStore(filepath)
    root = zarr.group(store=store, overwrite=True)

    root.create_dataset("reference_tmp", shape=(height, width, channels), fill_value=0, chunks=True, dtype="float32")
    root.create_dataset("count_tmp", shape=(height, width, channels), fill_value=0, chunks=True, dtype="uint16")

    final_zarr = root.create_dataset("data", shape=(height, width, channels), fill_value=0, chunks=True, dtype="float32")

    # Add the metadata to the zarr file
    final_zarr.attrs["_ARRAY_DIMENSIONS"] = ["x", "y", "z"]
    final_zarr.attrs['width'] = width
    final_zarr.attrs['height'] = height
    final_zarr.attrs['count'] = channels
    final_zarr.attrs['dtype'] = "float32"
    final_zarr.attrs['bounds'] = bounds
    final_zarr.attrs['transform'] = transform
    final_zarr.attrs['crs'] = "EPSG:2056"

    return root, transform
    

def project_patch(lon_left, lon_right, lat_bottom, lat_top, patch, group_zarr, transform, subraster_margin=3, nx=128, ny=128):
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
    patch_values = patch.reshape(-1, patch.shape[2])
    
    # Determine the bounding box for the subraster
    min_y_st, min_x_st = np.min(y_st), np.min(x_st)
    max_y_st, max_x_st = np.max(y_st), np.max(x_st)

    # Determine the corresponding indices in the reference raster
    min_j_dst, max_i_dst = ~transform * (min_x_st, min_y_st)
    max_j_dst, min_i_dst = ~transform * (max_x_st, max_y_st)

    # Define a margin around the patch for the subraster
    min_j, min_i = int(min_j_dst) - subraster_margin, int(min_i_dst) - subraster_margin
    max_j, max_i = int(max_j_dst) + subraster_margin, int(max_i_dst) + subraster_margin
    
    # Extract the subraster and corresponding coordinates
    reference_raster, count_raster = group_zarr["reference_tmp"], group_zarr["count_tmp"]
    subraster = reference_raster[min_i:max_i, min_j:max_j, :]
    subraster_count = count_raster[min_i:max_i, min_j:max_j, :]

    sub_y, sub_x = np.mgrid[min_i:max_i, min_j:max_j]
    sub_lon, sub_lat = rasterio.transform.xy(transform, sub_y, sub_x, offset='center')
    dst_points = np.array([np.ravel(sub_lat), np.ravel(sub_lon)]).T

    # Interpolate the patch data to fit the reference raster grid
    for i in range(patch.shape[2]):
        resampled_patch = griddata(src_points, patch_values[:, i], dst_points, method='linear').reshape(subraster.shape[:2])
        # Add resampled patch to reference raster
        subraster[..., i] += np.nan_to_num(resampled_patch)
        subraster_count[..., i] += ~np.isnan(resampled_patch)

    # Place the subraster back into the reference raster
    reference_raster[min_i:max_i, min_j:max_j, :] = subraster
    count_raster[min_i:max_i, min_j:max_j, :] = subraster_count


def map_patches_to_raster(patches, coords, group_zarr, transform):
    """
    Map patches onto the reference raster.
    """
    for patch, c in zip(patches, coords):
        project_patch(c[0], c[1], c[2], c[3], patch, group_zarr, transform)


def correct_overlap(group_zarr):

    raster, count, final = group_zarr["reference_tmp"], group_zarr["count_tmp"], group_zarr["data"]

    for c in range(final.attrs["count"]):
        final[:, :, c] = np.divide(raster[:, :, c], count[:, :, c], out=np.full((final.attrs["height"], final.attrs["width"]), fill_value=np.nan), where=count[:, :, c] != 0)

    del group_zarr["reference_tmp"]
    del group_zarr["count_tmp"]

    zarr.consolidate_metadata(group_zarr.store)


if __name__ == "__main__":

    COORDS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/coords_nolon_era_500k.npy"
    PARAMS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/params_nolon_era_500k.npy"
    ANOMS_FILENAME = "/data_2/scratch/dbrueggemann/nn/overlay/anoms_nolon_era_500k.npy"

    out_path = "/data_2/scratch/dbrueggemann/nn/overlay"

    # output of `extract_maps.py`
    # N is the number of cubes
    # coords contains: (lon_left, lon_right, lat_bottom, lat_top) for each cube
    coords = np.load(COORDS_FILENAME)  # N x 4
    
    # Parameter predictions first
    print("Doing parameters")
    params = np.load(PARAMS_FILENAME)  # N x 128 x 128 x 6
    params_filename = os.path.join(out_path, "parameters.zarr")
    group_zarr, transform = create_reference_raster(filepath=params_filename, channels=params.shape[3])
    map_patches_to_raster(params, coords, group_zarr, transform)
    correct_overlap(group_zarr)
    del params

    # Anomalies next
    print("Doing anomalies")
    anoms = np.load(ANOMS_FILENAME)
    anoms = anoms.transpose(0, 2, 3, 1)  # N x 128 x 128 x 73
    anoms_filename = os.path.join(out_path, "anomalies.zarr")
    group_zarr, transform = create_reference_raster(filepath=anoms_filename, channels=anoms.shape[3])
    map_patches_to_raster(anoms, coords, group_zarr, transform)
    correct_overlap(group_zarr)
