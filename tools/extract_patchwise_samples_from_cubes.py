import glob
import os
import argparse

import h5py
import numpy as np
import xarray as xr

SPLIT = "train"
USE_RAW = False
FOREST_THRESH = 0.8  # threshold of forest to consider to sample pixel
DROUGHT_THRESH = 0.0  # threshold of drought to consider to sample pixel
H = 128
W = 128
T = 73


def create_h5(file, key, data, shape, dtype):
    file.create_dataset(
        key,
        data=data,
        maxshape=(None, *shape),
        dtype=dtype,
        compression="gzip",
        compression_opts=9,
    )


def append_h5(file, key, data):
    file[key].resize((file[key].shape[0] + data.shape[0]), axis=0)
    file[key][-data.shape[0] :] = data


def save_to_h5(
    h5_file,
    cube_chunk,
):
    """
    Saves cube_chunk to hdf5 file.
    """
    # TODO: create train/val/test split
    # TODO: double check data types --> compression

    valid_mask = (
        cube_chunk.FOREST_MASK.values > FOREST_THRESH
    ) & cube_chunk.to_sample.values.astype(bool)
    # check if any valid pixels in patch
    if valid_mask.any():
        s2_b02 = np.transpose(
            np.array(cube_chunk.s2_B02.values, dtype=np.float32)[np.newaxis, ...],
            (0, 2, 3, 1),
        )  # shape: 1 x H x W x T
        s2_b03 = np.transpose(
            np.array(cube_chunk.s2_B03.values, dtype=np.float32)[np.newaxis, ...],
            (0, 2, 3, 1),
        )  # shape: 1 x H x W x T
        s2_b04 = np.transpose(
            np.array(cube_chunk.s2_B04.values, dtype=np.float32)[np.newaxis, ...],
            (0, 2, 3, 1),
        )  # shape: 1 x H x W x T
        s2_b08 = np.transpose(
            np.array(cube_chunk.s2_B08.values, dtype=np.float32)[np.newaxis, ...],
            (0, 2, 3, 1),
        )  # shape: 1 x H x W x T
        s2_ndvi = np.transpose(
            np.array(cube_chunk.s2_ndvi.values, dtype=np.float32)[np.newaxis, ...],
            (0, 2, 3, 1),
        )  # shape: 1 x H x W x T
        slope = np.array(cube_chunk.slope.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        easting = np.array(cube_chunk.easting.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        twi = np.array(cube_chunk.twi.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        time = np.array(cube_chunk.time.values, dtype="S29")[
            np.newaxis, ...
        ]  # shape: 1 x T
        longitude = np.array(cube_chunk.lon.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x W
        latitude = np.array(cube_chunk.lat.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H
        drought_mask = np.array(
            cube_chunk.DROUGHT_MASK.values > DROUGHT_THRESH, dtype=bool
        )[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        height_idx, width_idx = np.indices(valid_mask.shape)
        sel_height_idx = height_idx[valid_mask]
        sel_width_idx = width_idx[valid_mask]
        sel_img_idx = np.zeros_like(sel_height_idx)
        valid_pixel_idx = np.stack(
            (sel_img_idx, sel_height_idx, sel_width_idx), axis=1
        )  # shape: N x 3
        valid_mask = np.array(valid_mask, dtype=bool)[
            np.newaxis, ...
        ]  # shape: 1 x H x W

        if not "spatiotemporal" in h5_file.keys():
            create_h5(h5_file, "spatiotemporal/s2_B02", s2_b02, (H, W, T), "float32")
            create_h5(h5_file, "spatiotemporal/s2_B03", s2_b03, (H, W, T), "float32")
            create_h5(h5_file, "spatiotemporal/s2_B04", s2_b04, (H, W, T), "float32")
            create_h5(h5_file, "spatiotemporal/s2_B08", s2_b08, (H, W, T), "float32")
            create_h5(h5_file, "spatiotemporal/s2_ndvi", s2_ndvi, (H, W, T), "float32")
            create_h5(h5_file, "spatial/slope", slope, (H, W), "float32")
            create_h5(h5_file, "spatial/easting", easting, (H, W), "float32")
            create_h5(h5_file, "spatial/twi", twi, (H, W), "float32")
            create_h5(h5_file, "spatial/drought_mask", drought_mask, (H, W), "bool")
            create_h5(h5_file, "spatial/valid_mask", valid_mask, (H, W), "bool")
            create_h5(h5_file, "temporal/time", time, (T,), "S29")
            create_h5(
                h5_file, "meta/longitude", longitude, (longitude.shape[1],), "float32"
            )
            create_h5(
                h5_file, "meta/latitude", latitude, (latitude.shape[1],), "float32"
            )
            create_h5(
                h5_file,
                "meta/valid_pixel_idx",
                valid_pixel_idx,
                (valid_pixel_idx.shape[1],),
                "uint16",
            )
        else:
            valid_pixel_idx[:, 0] += h5_file["spatiotemporal/s2_B02"].shape[0]
            append_h5(h5_file, "spatiotemporal/s2_B02", s2_b02)
            append_h5(h5_file, "spatiotemporal/s2_B03", s2_b03)
            append_h5(h5_file, "spatiotemporal/s2_B04", s2_b04)
            append_h5(h5_file, "spatiotemporal/s2_B08", s2_b08)
            append_h5(h5_file, "spatiotemporal/s2_ndvi", s2_ndvi)
            append_h5(h5_file, "spatial/slope", slope)
            append_h5(h5_file, "spatial/easting", easting)
            append_h5(h5_file, "spatial/twi", twi)
            append_h5(h5_file, "spatial/drought_mask", drought_mask)
            append_h5(h5_file, "spatial/valid_mask", valid_mask)
            append_h5(h5_file, "temporal/time", time)
            append_h5(h5_file, "meta/longitude", longitude)
            append_h5(h5_file, "meta/latitude", latitude)
            append_h5(h5_file, "meta/valid_pixel_idx", valid_pixel_idx)


def extract_samples_from_cubes(root_dir):
    """
    Generate h5 file for a split.
    """

    if USE_RAW:
        search_cube = glob.glob(os.path.join(root_dir, "cubes", "*_raw.nc"))
    else:
        search_cube = glob.glob(os.path.join(root_dir, "cubes", "*[!_raw].nc"))

    with h5py.File(os.path.join(root_dir, "tmp3_{}.h5".format(SPLIT)), "a") as h5_file:

        for cube_name in search_cube:
            cube = xr.open_dataset(os.path.join(root_dir, cube_name), engine="h5netcdf")
            print("Generating samples from loaded cube {}...".format(cube_name))
            # only consider complete years (January - December)
            if cube.time.dt.month[0] == 1 and cube.time.dt.day[0] <= 5:
                start_year = cube.time.dt.year[0].values
            else:
                start_year = cube.time.dt.year[0].values + 1
            if cube.time.dt.month[-1] == 12 and cube.time.dt.day[-1] >= 27:
                end_year = cube.time.dt.year[-1].values
            else:
                end_year = cube.time.dt.year[-1].values - 1

            for year in range(start_year, end_year + 1):
                cube_chunk = cube.sel(
                    time=slice("{}-01-01".format(year), "{}-12-31".format(year))
                )
                save_to_h5(
                    h5_file,
                    cube_chunk,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patchwise samples from cubes")
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory containing the datacubes",
    )
    args = parser.parse_args()

    extract_samples_from_cubes(args.root_dir)
