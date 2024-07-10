import argparse
import glob
import os
import time

import h5py
import numpy as np
import xarray as xr

from data.cloud_cleaning import smooth_s2_timeseries

# SPLIT = "train"
FOREST_THRESH = 0.8  # threshold of forest to consider to sample pixel
CLOUD_CLEANING = True
MAX_NAN = 36
REMOVE_PCT = 0.0  #  0.05
SMOOTHER = "lowess"
LOWESS_FRAC = 0.07
SG_WINDOW_LENGTH = 15
SG_POLYORDER = 2
SAVE_RAW = False
# DROUGHT_THRESH = 0.0  # threshold of drought to consider to sample pixel
H = 128
W = 128
START_YEAR = 2017
END_YEAR = 2023
T = (END_YEAR - START_YEAR + 1) * 73
ST_CHUNK_SIZE = (1, T, 4, 4)
S_CHUNK_SIZE = (1, H, W)


def create_h5(file, key, data, shape, dtype, chunk_size=None):
    file.create_dataset(
        key,
        data=data,
        maxshape=(None, *shape),
        dtype=dtype,
        compression="lzf",
        chunks=chunk_size,
    )


def append_h5(file, key, data):
    file[key].resize((file[key].shape[0] + data.shape[0]), axis=0)
    file[key][-data.shape[0] :] = data


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


def save_to_h5(
    h5_file,
    cube,
):
    """
    Saves cube_chunk to hdf5 file.
    """
    # TODO: create train/val/test split
    # TODO: double check data types --> compression
    # TODO: auto-chunking is enabled due to compression, adapt chunk size?

    missing_dates = check_missing_timestamps(cube)
    if missing_dates:
        print(f"Inserting missing timestamps: {missing_dates}")
        cube = cube.reindex(
            time=np.sort(np.concatenate([cube.time.values, missing_dates]))
        )
        if not CLOUD_CLEANING:
            cube["s2_ndvi"] = cube["s2_ndvi"].interpolate_na(
                dim="time", method="linear"
            )

    # Cloud cleaning
    if CLOUD_CLEANING:
        cube = smooth_s2_timeseries(
            cube,
            MAX_NAN,
            REMOVE_PCT,
            SMOOTHER,
            LOWESS_FRAC,
            SG_WINDOW_LENGTH,
            SG_POLYORDER,
        )
        print("Performed cloud cleaning")
    else:
        # Set to_sample = 1 everywhere if cloud cleaning is not performed
        ones_dataarray = xr.DataArray(
            np.ones((len(cube.lat), len(cube.lon))), dims=("lat", "lon")
        )
        cube["to_sample"] = ones_dataarray

    # Deal with NaNs (or -9999)
    # For era5 linear interpolation
    cube_tmp = cube[[name for name in list(cube.variables) if name.startswith("era5")]]
    if np.isnan(cube_tmp.to_array()).any():
        cube[[name for name in list(cube.variables) if name.startswith("era5")]] = cube[
            [name for name in list(cube.variables) if name.startswith("era5")]
        ].interpolate_na(dim="time", method="linear")
    # For static layers to average in space
    variables_to_fill = [
        name
        for name in list(cube.variables)
        if not name.startswith("era5")
        and not name.startswith("s2")
        and name not in ["time", "lat", "lon", "to_sample"]
    ]
    cube_tmp = cube[variables_to_fill]
    cube_tmp = cube_tmp.where(cube_tmp != -9999, np.nan)
    if np.isnan(cube_tmp.to_array()).any():
        mean = cube_tmp[variables_to_fill].mean().to_array().values
        for i, v in enumerate(variables_to_fill):
            cube[v].fillna(mean[i])
    print("Dealt with missing values")

    valid_mask = (
        cube.FOREST_MASK.values > FOREST_THRESH
    ) & cube.to_sample.values.astype(bool)
    # check if any valid pixels in patch
    if valid_mask.any():
        s2_b02 = np.array(cube.s2_B02.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x T x H x W
        s2_b03 = np.array(cube.s2_B03.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x T x H x W
        s2_b04 = np.array(cube.s2_B04.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x T x H x W
        s2_b08 = np.array(cube.s2_B08.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x T x H x W
        s2_ndvi = np.array(cube.s2_ndvi.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x T x H x W
        if CLOUD_CLEANING and SAVE_RAW:
            s2_raw_ndvi = np.array(cube.s2_raw_ndvi.values, dtype=np.float32)[
                np.newaxis, ...
            ]  # shape: 1 x T x H x W
            s2_cloud_cleaned_ndvi = np.array(
                cube.s2_cloud_cleaned_ndvi.values, dtype=np.float32
            )[
                np.newaxis, ...
            ]  # shape: 1 x T x H x W
        slope = np.array(cube.slope.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        easting = np.array(cube.easting.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        twi = np.array(cube.twi.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        time = np.array(cube.time.values, dtype="S29")[np.newaxis, ...]  # shape: 1 x T
        longitude = np.array(cube.lon.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x W
        latitude = np.array(cube.lat.values, dtype=np.float32)[
            np.newaxis, ...
        ]  # shape: 1 x H
        drought_mask = np.array(cube.DROUGHT_MASK.values)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        drought_mask[drought_mask == -9999] = 255
        drought_mask = drought_mask.astype(np.uint8)
        height_idx, width_idx = np.indices(valid_mask.shape)
        sel_height_idx = height_idx[valid_mask]
        sel_width_idx = width_idx[valid_mask]
        sel_img_idx = np.zeros_like(sel_height_idx)
        pixel_idx = np.stack(
            (sel_img_idx, sel_height_idx, sel_width_idx), axis=1
        )  # shape: N x 3
        valid_mask = np.array(valid_mask, dtype=bool)[
            np.newaxis, ...
        ]  # shape: 1 x H x W
        # check how many full years
        if cube.time.dt.month[0] == 1 and cube.time.dt.day[0] <= 5:
            start_year = cube.time.dt.year[0].values
        else:
            start_year = cube.time.dt.year[0].values + 1
        if cube.time.dt.month[-1] == 12 and cube.time.dt.day[-1] >= 27:
            end_year = cube.time.dt.year[-1].values
        else:
            end_year = cube.time.dt.year[-1].values - 1
        min_t_idx = [
            np.min(np.argwhere(cube.time.dt.year.values == y))
            for y in range(start_year, end_year + 1)
        ]
        max_t_idx = [
            np.max(np.argwhere(cube.time.dt.year.values == y))
            for y in range(start_year, end_year + 1)
        ]

        # for each pixel_idx, unfold each year
        expanded_pixel_idx = np.repeat(pixel_idx, len(min_t_idx), axis=0)
        expanded_min_t_idx = np.tile(min_t_idx, len(pixel_idx))[:, np.newaxis]
        expanded_max_t_idx = np.tile(max_t_idx, len(pixel_idx))[:, np.newaxis]
        annual_pixel_idx = np.concatenate(
            (expanded_pixel_idx, expanded_min_t_idx, expanded_max_t_idx), axis=1
        )  # YN x 5
        sel_img_idx = np.zeros_like(min_t_idx)
        annual_idx = np.stack((sel_img_idx, min_t_idx, max_t_idx), axis=1)  # Y x 3

        if not "spatiotemporal" in h5_file.keys():
            create_h5(
                h5_file,
                "spatiotemporal/s2_B02",
                s2_b02,
                (T, H, W),
                "float32",
                ST_CHUNK_SIZE,
            )
            create_h5(
                h5_file,
                "spatiotemporal/s2_B03",
                s2_b03,
                (T, H, W),
                "float32",
                ST_CHUNK_SIZE,
            )
            create_h5(
                h5_file,
                "spatiotemporal/s2_B04",
                s2_b04,
                (T, H, W),
                "float32",
                ST_CHUNK_SIZE,
            )
            create_h5(
                h5_file,
                "spatiotemporal/s2_B08",
                s2_b08,
                (T, H, W),
                "float32",
                ST_CHUNK_SIZE,
            )
            create_h5(
                h5_file,
                "spatiotemporal/s2_ndvi",
                s2_ndvi,
                (T, H, W),
                "float32",
                ST_CHUNK_SIZE,
            )
            if CLOUD_CLEANING and SAVE_RAW:
                create_h5(
                    h5_file,
                    "spatiotemporal/s2_raw_ndvi",
                    s2_raw_ndvi,
                    (T, H, W),
                    "float32",
                    ST_CHUNK_SIZE,
                )
                create_h5(
                    h5_file,
                    "spatiotemporal/s2_cloud_cleaned_ndvi",
                    s2_cloud_cleaned_ndvi,
                    (T, H, W),
                    "float32",
                    ST_CHUNK_SIZE,
                )
            create_h5(h5_file, "spatial/slope", slope, (H, W), "float32", S_CHUNK_SIZE)
            create_h5(
                h5_file, "spatial/easting", easting, (H, W), "float32", S_CHUNK_SIZE
            )
            create_h5(h5_file, "spatial/twi", twi, (H, W), "float32", S_CHUNK_SIZE)
            create_h5(
                h5_file,
                "spatial/drought_mask",
                drought_mask,
                (H, W),
                "uint8",
                S_CHUNK_SIZE,
            )
            create_h5(
                h5_file, "spatial/valid_mask", valid_mask, (H, W), "bool", S_CHUNK_SIZE
            )
            create_h5(h5_file, "temporal/time", time, (T,), "S29", (1, T))
            create_h5(
                h5_file, "meta/longitude", longitude, (longitude.shape[1],), "float32"
            )
            create_h5(
                h5_file, "meta/latitude", latitude, (latitude.shape[1],), "float32"
            )
            create_h5(
                h5_file,
                "meta/pixel_idx",
                pixel_idx,
                (pixel_idx.shape[1],),
                "uint16",
            )
            create_h5(
                h5_file,
                "meta/annual_idx",
                annual_idx,
                (annual_idx.shape[1],),
                "uint16",
            )
            create_h5(
                h5_file,
                "meta/annual_pixel_idx",
                annual_pixel_idx,
                (annual_pixel_idx.shape[1],),
                "uint16",
            )
        else:
            pixel_idx[:, 0] += h5_file["spatiotemporal/s2_B02"].shape[0]
            annual_idx[:, 0] += h5_file["spatiotemporal/s2_B02"].shape[0]
            annual_pixel_idx[:, 0] += h5_file["spatiotemporal/s2_B02"].shape[0]
            append_h5(h5_file, "spatiotemporal/s2_B02", s2_b02)
            append_h5(h5_file, "spatiotemporal/s2_B03", s2_b03)
            append_h5(h5_file, "spatiotemporal/s2_B04", s2_b04)
            append_h5(h5_file, "spatiotemporal/s2_B08", s2_b08)
            append_h5(h5_file, "spatiotemporal/s2_ndvi", s2_ndvi)
            if CLOUD_CLEANING and SAVE_RAW:
                append_h5(h5_file, "spatiotemporal/s2_raw_ndvi", s2_raw_ndvi)
                append_h5(
                    h5_file,
                    "spatiotemporal/s2_cloud_cleaned_ndvi",
                    s2_cloud_cleaned_ndvi,
                )
            append_h5(h5_file, "spatial/slope", slope)
            append_h5(h5_file, "spatial/easting", easting)
            append_h5(h5_file, "spatial/twi", twi)
            append_h5(h5_file, "spatial/drought_mask", drought_mask)
            append_h5(h5_file, "spatial/valid_mask", valid_mask)
            append_h5(h5_file, "temporal/time", time)
            append_h5(h5_file, "meta/longitude", longitude)
            append_h5(h5_file, "meta/latitude", latitude)
            append_h5(h5_file, "meta/pixel_idx", pixel_idx)
            append_h5(h5_file, "meta/annual_idx", annual_idx)
            append_h5(h5_file, "meta/annual_pixel_idx", annual_pixel_idx)


def extract_samples_from_cubes(data_dir, save_dir):
    """
    Generate h5 file for a split.
    """
    if CLOUD_CLEANING:
        search_cube = glob.glob(os.path.join(data_dir, "cubes", "*_raw.nc"))
    else:
        search_cube = glob.glob(os.path.join(data_dir, "cubes", "*.nc"))

    extracted_filename = os.path.join(save_dir, "extracted_maxnan{}_removepct{}_lowessfrac{}.txt".format(MAX_NAN, REMOVE_PCT, LOWESS_FRAC))
    if os.path.isfile(extracted_filename):
        with open(extracted_filename, "r") as f:
            extracted = f.read().splitlines()
    else:
        extracted = []

    if os.path.isfile(os.path.join(save_dir, "failed_cubes.txt")):
        with open(os.path.join(save_dir, "failed_cubes.txt"), "r") as f:
            extracted.extend(f.read().splitlines())

    with h5py.File(
        os.path.join(
            save_dir,
            "processed_maxnan{}_removepct{}_lowessfrac{}.h5".format(
                MAX_NAN, REMOVE_PCT, LOWESS_FRAC
            ),
        ),
        "a",
    ) as h5_file:

        for cube_name in search_cube:
            if os.path.basename(cube_name) in extracted:
                print("Already extracted {}, skipping...".format(cube_name))
                continue

            # if os.path.basename(cube_name) in [  # invalid list
            #     "2017_1_30_2023_12_30_8.532724933477382_46.35084937056722_128_128_raw.nc"
            # ]:
            #     continue

            start = time.time()
            try:
                cube = xr.open_dataset(
                    os.path.join(data_dir, cube_name),
                    engine="h5netcdf",
                )
            except OSError:  # this happens when the cube is corrupt
                with open(os.path.join(save_dir, "failed_cubes.txt"), "a") as f:
                    f.write(os.path.basename(cube_name) + '\n')
                print("failed to read cube {}, skipping...".format(cube_name))
                continue

            print("Generating samples from loaded cube {}...".format(cube_name))
            try:
                save_to_h5(
                    h5_file,
                    cube,
                )
            except KeyError:  # this happens when variables are missing
                with open(os.path.join(save_dir, "failed_cubes.txt"), "a") as f:
                    f.write(os.path.basename(cube_name) + '\n')
                print("could not find variables in cube {}, skipping...".format(cube_name))
                continue
            end = time.time()
            print("time: ", end - start)

            with open(os.path.join(save_dir, "extracted_maxnan{}_removepct{}_lowessfrac{}.txt".format(
                        MAX_NAN, REMOVE_PCT, LOWESS_FRAC
                    )), "a") as f:
                f.write(os.path.basename(cube_name) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patchwise samples from cubes")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the datacubes",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save h5",
    )
    args = parser.parse_args()

    extract_samples_from_cubes(args.data_dir, args.save_dir)
