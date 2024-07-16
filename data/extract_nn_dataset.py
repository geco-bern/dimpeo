import xarray as xr
import glob
import os
import h5py
import numpy as np
import itertools

from utils.helpers import check_missing_timestamps, get_doy

H = 128
W = 128
START_YEAR = 2017
END_YEAR = 2023
T = (END_YEAR - START_YEAR + 1) * 73

save_dir = "/data_1/scratch_1/dbrueggemann"
cubes = glob.glob("/data_2/dimpeo/cubes/*_raw.nc")

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


with h5py.File(os.path.join(save_dir, "qforest_dataset_full.h5"), "w") as h5_file:
    for i, c in enumerate(cubes):

        try:
            minicube = xr.open_dataset(c, engine="h5netcdf")
        except OSError:
            continue

        missing_dates = check_missing_timestamps(minicube, START_YEAR, END_YEAR)
        if missing_dates:
            minicube = minicube.reindex(
                time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
            )

        try:
            s2_cube = minicube.s2_ndvi.where((minicube.s2_mask == 0) & minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7])).values
        except AttributeError:
            continue
        s2_mask = (minicube.FOREST_MASK.values > 0.8)
        
        pixels = s2_cube[:, s2_mask].transpose(1, 0)

        longitude = np.array(minicube.lon.values, dtype=np.float32)
        latitude = np.array(minicube.lat.values, dtype=np.float32)
        lon_lat = cartesian_product(longitude, latitude).reshape(len(longitude), len(latitude), 2)
        lon_lat = lon_lat[s2_mask, :]

        dem = np.expand_dims(minicube.DEM.values[s2_mask], axis=1)
        fc = np.expand_dims(minicube.FC.values[s2_mask], axis=1)
        fh = np.expand_dims(minicube.FH.values[s2_mask], axis=1)
        slope = np.expand_dims(minicube.slope.values[s2_mask], axis=1)
        easting = np.expand_dims(minicube.easting.values[s2_mask], axis=1)
        northing = np.expand_dims(minicube.northing.values[s2_mask], axis=1)
        twi = np.expand_dims(minicube.twi.values[s2_mask], axis=1)

        N = pixels.shape[0]

        doy = np.expand_dims(get_doy(minicube.time.values), axis=0).repeat(N, axis=0)

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
                "doy",
                doy,
                (T,),
                "uint16",
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
            append_h5(h5_file, "doy", doy)

        if i % 100 == 0:
            print("Done {} cubes".format(i))
