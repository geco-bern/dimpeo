import xarray as xr
import glob
import os
import h5py
import numpy as np
import itertools

from neural_network.helpers import (
    check_missing_timestamps,
    get_doy,
    START_YEAR,
    END_YEAR,
    NUM_DATAPOINTS_PER_YEAR,
)


T = (END_YEAR - START_YEAR + 1) * NUM_DATAPOINTS_PER_YEAR


def cartesian_product(a, b):
    return np.array(list(itertools.product(a, b)))


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


def extract_dataset(save_dir, cubes):
    with h5py.File(os.path.join(save_dir, "nn_dataset.h5"), "w") as h5_file:
        for i, c in enumerate(cubes):

            try:
                minicube = xr.open_dataset(c, engine="h5netcdf")
            except OSError:
                with open(os.path.join(save_dir, "failed_cubes.txt"), "a") as f:
                    f.write(os.path.basename(c) + "\n")
                continue

            missing_dates = check_missing_timestamps(minicube)
            if missing_dates:
                minicube = minicube.reindex(
                    time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
                )

            try:
                s2_cube = minicube.s2_ndvi.where(
                    (minicube.s2_mask == 0) & minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7])
                ).values
            except AttributeError:
                with open(os.path.join(save_dir, "failed_cubes.txt"), "a") as f:
                    f.write(os.path.basename(c) + "\n")
                continue
            s2_mask = minicube.FOREST_MASK.values > 0.8

            pixels = s2_cube[:, s2_mask].transpose(1, 0)

            longitude = np.array(minicube.lon.values, dtype=np.float32)
            latitude = np.array(minicube.lat.values, dtype=np.float32)
            lon_lat = cartesian_product(longitude, latitude).reshape(
                len(longitude), len(latitude), 2
            )
            lon_lat = lon_lat[s2_mask, :]

            dem = np.expand_dims(minicube.DEM.values[s2_mask], axis=1)
            fc = np.expand_dims(minicube.FC.values[s2_mask], axis=1)
            fh = np.expand_dims(minicube.FH.values[s2_mask], axis=1)
            slope = np.expand_dims(minicube.slope.values[s2_mask], axis=1)
            easting = np.expand_dims(minicube.easting.values[s2_mask], axis=1)
            northing = np.expand_dims(minicube.northing.values[s2_mask], axis=1)
            twi = np.expand_dims(minicube.twi.values[s2_mask], axis=1)
            rugg = np.expand_dims(minicube.rugg.values[s2_mask], axis=1)
            curv = np.expand_dims(minicube.curv.values[s2_mask], axis=1)

            # we need to take the mean and std without biasing it towards missing value
            # get first a mean annual curve, then take mean and std
            pressure = minicube.era5_sp.values[:, s2_mask]
            annual_pressure = np.nanmean(
                np.reshape(
                    pressure, (END_YEAR - START_YEAR + 1, NUM_DATAPOINTS_PER_YEAR, -1)
                ),
                axis=0,
            )
            pressure_mean = np.expand_dims(np.mean(annual_pressure, axis=0), axis=1)
            pressure_std = np.expand_dims(np.std(annual_pressure, axis=0), axis=1)

            temperature = minicube.era5_t2m.values[:, s2_mask]
            annual_temperature = np.nanmean(
                np.reshape(
                    temperature,
                    (END_YEAR - START_YEAR + 1, NUM_DATAPOINTS_PER_YEAR, -1),
                ),
                axis=0,
            )
            temperature_mean = np.expand_dims(
                np.mean(annual_temperature, axis=0), axis=1
            )
            temperature_std = np.expand_dims(np.std(annual_temperature, axis=0), axis=1)

            precipitation = minicube.era5_tp.values[:, s2_mask]
            annual_precipitation = np.nanmean(
                np.reshape(
                    precipitation,
                    (END_YEAR - START_YEAR + 1, NUM_DATAPOINTS_PER_YEAR, -1),
                ),
                axis=0,
            )
            precipitation_mean = np.expand_dims(
                np.mean(annual_precipitation, axis=0), axis=1
            )
            precipitation_std = np.expand_dims(
                np.std(annual_precipitation, axis=0), axis=1
            )

            N = pixels.shape[0]

            doy = np.expand_dims(get_doy(minicube.time.values), axis=0).repeat(
                N, axis=0
            )

            if not "ndvi" in h5_file.keys():
                create_h5(
                    h5_file,
                    "ndvi",
                    pixels,
                    (T,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "lon_lat",
                    lon_lat,
                    (2,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "dem",
                    dem,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "fc",
                    fc,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "fh",
                    fh,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "slope",
                    slope,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "easting",
                    easting,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "northing",
                    northing,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "twi",
                    twi,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "rugg",
                    rugg,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "curv",
                    curv,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "doy",
                    doy,
                    (T,),
                    "uint16",
                )
                create_h5(
                    h5_file,
                    "press_mean",
                    pressure_mean,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "press_std",
                    pressure_std,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "temp_mean",
                    temperature_mean,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "temp_std",
                    temperature_std,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "precip_mean",
                    precipitation_mean,
                    (1,),
                    "float32",
                )
                create_h5(
                    h5_file,
                    "precip_std",
                    precipitation_std,
                    (1,),
                    "float32",
                )
            else:
                append_h5(h5_file, "ndvi", pixels)
                append_h5(h5_file, "lon_lat", lon_lat)
                append_h5(h5_file, "dem", dem)
                append_h5(h5_file, "fc", fc)
                append_h5(h5_file, "fh", fh)
                append_h5(h5_file, "slope", slope)
                append_h5(h5_file, "easting", easting)
                append_h5(h5_file, "northing", northing)
                append_h5(h5_file, "twi", twi)
                append_h5(h5_file, "rugg", rugg)
                append_h5(h5_file, "curv", curv)
                append_h5(h5_file, "doy", doy)
                append_h5(h5_file, "press_mean", pressure_mean)
                append_h5(h5_file, "press_std", pressure_std)
                append_h5(h5_file, "temp_mean", temperature_mean)
                append_h5(h5_file, "temp_std", temperature_std)
                append_h5(h5_file, "precip_mean", precipitation_mean)
                append_h5(h5_file, "precip_std", precipitation_std)

            if i % 100 == 0:
                print("Done {} cubes".format(i))


def compute_missingness(save_dir):
    with h5py.File(os.path.join(save_dir, "nn_dataset.h5"), "r") as file:
        ndvi = file.get("ndvi")[:]

    spl = np.concatenate(np.split(ndvi, (END_YEAR - START_YEAR + 1), axis=1), axis=0)
    missingness = np.isnan(spl).sum(axis=0) / spl.shape[0]
    np.save(os.path.join(save_dir, "missingness.py"), missingness)


if __name__ == "__main__":

    save_dir = os.environ["SAVE_DIR"]
    cubes = glob.glob(os.path.join(os.environ["CUBE_DIR"], "*_raw.nc"))
    extract_dataset(save_dir, cubes)
    compute_missingness(save_dir)
