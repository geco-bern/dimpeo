import glob
import os

import h5py
import numpy as np
import xarray as xr

CONFIG = {
    # TODO: export more variables
    "variables_to_export": [
        "s2_B02",
        "s2_B03",
        "s2_B04",
        "s2_B08",
        "s2_ndvi",
        "slope",
        "DROUGHT_MASK",
    ],
    # Which split
    "split": "train",  # split name
    # Sample format
    "start_date": "01-01",  # in mm-dd
    "end_date": "12-31",  # in mm-dd
    "use_raw": False,
    # Samples criteria
    "drought_labels": False,  # If to use drought mask for sampling pixels
    "forest_thresh": 0.8,  # threshold of forest to consider to sample pixel
    "drought_thresh": 0,  # threshold of drought to consider to sample pixel
}


def save_pixel_timeseries(h5_file, cube_data, time, longitude, latitude):
    """
    Save cube to split directory. Will create the directories if not existing yet

    :param h5_file: h5 file handle to save time-series in
    :param cube_data: data cube to save
    :param time:
    :param longitude:
    :param latitude:
    """
    # TODO: export variable-length string with variables names (meta data)
    # TODO: save data in chunks?
    cube_data = np.transpose(
        np.array(cube_data, dtype=np.float32)[np.newaxis, ...], (0, 3, 4, 2, 1)
    )  # shape is 1 x H x W x T x C
    time = np.array(time, dtype="S29")[np.newaxis, ...]
    longitude = np.array(longitude, dtype=np.float32)[np.newaxis, ...]
    latitude = np.array(latitude, dtype=np.float32)[np.newaxis, ...]

    if not "data" in h5_file.keys():
        h5_file.create_dataset(
            "data",
            data=cube_data,
            maxshape=(None, *cube_data.shape[1:]),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        h5_file.create_dataset(
            "time",
            data=time,
            maxshape=(None, time.shape[1]),
            dtype="S29",
            compression="gzip",
            compression_opts=9,
        )
        h5_file.create_dataset(
            "longitude",
            data=longitude,
            maxshape=(None, longitude.shape[1]),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
        h5_file.create_dataset(
            "latitude",
            data=latitude,
            maxshape=(None, latitude.shape[1]),
            dtype="float32",
            compression="gzip",
            compression_opts=9,
        )
    else:
        h5_file["data"].resize((h5_file["data"].shape[0] + cube_data.shape[0]), axis=0)
        h5_file["data"][-cube_data.shape[0] :] = cube_data
        h5_file["time"].resize((h5_file["time"].shape[0] + time.shape[0]), axis=0)
        h5_file["time"][-time.shape[0] :] = time
        h5_file["longitude"].resize(
            (h5_file["longitude"].shape[0] + longitude.shape[0]), axis=0
        )
        h5_file["longitude"][-longitude.shape[0] :] = longitude
        h5_file["latitude"].resize(
            (h5_file["latitude"].shape[0] + latitude.shape[0]), axis=0
        )
        h5_file["latitude"][-latitude.shape[0] :] = latitude


def extract_pixel_timeseries(
    h5_file,
    cube_chunk,
    split,
    variables_to_export,
    drought_labels,
    forest_thresh,
    drought_thresh,
):
    """
    From a datacube, will extract individual pixel timeseries based on forest mask and to_sample variable.
    Then save them as .npz files.

    :param h5_file: h5 file handle to save time-series in
    :param cube_chunk: xarray from which to extract data
    :param split: str, train/test/val or other name of split
    :param variables_to_export: variables to export when saving pixel timeseries
    :param forest_thresh: threshold to pass in sampling for forest ratio in pixel
    """
    # TODO: pick pixels according to split?
    # TODO: save drought labels

    check_forest_mask = cube_chunk.FOREST_MASK.values
    # check if any forest in patch
    if (check_forest_mask > forest_thresh).any():

        cube_data = cube_chunk[variables_to_export].to_array().values
        time = cube_chunk.time.values
        longitude = cube_chunk.lon.values
        latitude = cube_chunk.lat.values

        save_pixel_timeseries(
            h5_file,
            cube_data,
            time,
            longitude,
            latitude,
        )


def obtain_timeseries(
    cube,
    h5_file,
    start_date,
    end_date,
    split,
    variables_to_export,
    drought_labels=False,
    forest_thresh=0.5,
    drought_thresh=0,
):
    """
    Split into context target pairs given whole time interval of the cube

    :param h5_file: h5 file handle to save time-series in
    :param cube: data cube containing data across whole time interval
    :param start_date: int, index of starting month of time series
    :param end_date: int, index of ending month of time series
    :param split: str, train/test/val or other name of split
    :param variables_to_export: variables to export when saving pixel timeseries
    :param forest_thresh: threshold to pass in sampling for forest ratio in pixel
    :param drought_thresh: threshold to pass in sampling for drought ratio in pixel
    """

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

        sub_series = cube.sel(
            time=slice("{}-{}".format(year, start_date), "{}-{}".format(year, end_date))
        )

        # Extract pixels at lat lons here and save
        extract_pixel_timeseries(
            h5_file,
            sub_series,
            split,
            variables_to_export,
            drought_labels,
            forest_thresh,
            drought_thresh,
        )

    return


def extract_samples_from_cubes(root_dir):
    """
    Generate datacubes for a split, with a given context and target length and minicuber specifications.
    Save the cubes locally.

    Config containing
    :param split: str, split name
    :param normalisation: boolean. Compute min/max in space and time for each variables and save values
    :param cloud_cleaning: max number of consecutive nan value in timeseries allowed, after which cloud cleaning will be performed
    :param target_in_summer: If target start date should be included in June-Sep
    :param drought_labels: If to use drought mask for sampling pixels
    :param forest_thresh: threshold of forest to consider to sample pixel
    :param drought_thresh: threshold of drought to consider to sample pixel
    :param gen_samples: generate samples if cube already exists
    """

    if CONFIG["use_raw"]:
        search_cube = glob.glob(os.path.join(root_dir, "cubes", "*_raw.nc"))
    else:
        search_cube = glob.glob(os.path.join(root_dir, "cubes", "*[!_raw].nc"))

    with h5py.File(
        os.path.join(root_dir, "tmp2_{}.h5".format(CONFIG["split"])), "a"
    ) as h5_file:

        for cube_name in search_cube:
            cube = xr.open_dataset(os.path.join(root_dir, cube_name), engine="h5netcdf")
            print("Generating samples from loaded cube {}...".format(cube_name))
            obtain_timeseries(
                cube,
                h5_file,
                CONFIG["start_date"],
                CONFIG["end_date"],
                CONFIG["split"],
                CONFIG["variables_to_export"],
                CONFIG["drought_labels"],
                CONFIG["forest_thresh"],
                CONFIG["drought_thresh"],
            )


if __name__ == "__main__":
    root_dir = (
        "/Volumes/Macintosh HD/Users/davidbruggemann/OneDrive - epfl.ch/DIMPEO/data"
    )
    extract_samples_from_cubes(root_dir)
