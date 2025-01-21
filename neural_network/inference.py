import argparse
import numpy as np
import glob
import torch
import torch.nn as nn
import xarray as xr
import os

from neural_network.mlp import MLP
from neural_network.train import double_logistic_function, MEANS, STDS
from neural_network.helpers import get_doy, check_missing_timestamps, create_reference_raster, project_patch, consolidate, get_dates, convert_params


NUM_DATAPOINTS_PER_YEAR = 73
H, W = 128, 128
START_YEAR = 2017
END_YEAR = 2023
YEARS_IN_TRAIN = 6  # first six years in train, last year in test
SPLIT_IDX = NUM_DATAPOINTS_PER_YEAR * YEARS_IN_TRAIN
T_SCALE = 1.0 / 365.0


def rectify_parameters(params):
    # params: sos, mat_minus_sos, sen, eos_minus_sen, M, m
    inverted_mask = (params[:, 0] > params[:, 2])

    rec_params = params.clone()
    rec_params[inverted_mask, 0] = params[inverted_mask, 2]
    rec_params[inverted_mask, 1] = params[inverted_mask, 3]
    rec_params[inverted_mask, 2] = params[inverted_mask, 0]
    rec_params[inverted_mask, 3] = params[inverted_mask, 1]
    rec_params[inverted_mask, 4] = params[inverted_mask, 5] + params[inverted_mask, 5] - params[inverted_mask, 4]
    # rec_params[inverted_mask, 5] = params[inverted_mask, 5]

    return rec_params


def inference(name, encoder_name, features):

    anoms_path = os.path.join(os.environ["SAVE_DIR"], f"anomalies_{name}.zarr")
    params_path = os.path.join(os.environ["SAVE_DIR"], f"parameters_{name}.zarr")

    filepaths = sorted(list(glob.glob(os.path.join(os.environ["CUBE_DIR"], "*_raw.nc"))))
    device = "cuda"

    if features is None:
        features = ["dem", "fc", "fh", "slope", "easting", "northing", "twi", "rugg", "curv", "press_mean", "press_std", "temp_mean", "temp_std", "precip_mean", "precip_std"]

    encoder = MLP(d_in=len(features), d_out=8, n_blocks=8, d_block=256, dropout=0, skip_connection=True).to(device)
    encoder.load_state_dict(torch.load(os.path.join(os.environ["SAVE_DIR"], encoder_name)))
    encoder.eval()

    means = torch.tensor([MEANS[f] for f in features], dtype=float, device=device).unsqueeze(1).unsqueeze(1)
    stds = torch.tensor([STDS[f] for f in features], dtype=float, device=device).unsqueeze(1).unsqueeze(1)

    anom_dataset = create_reference_raster(filepath=anoms_path, channel_name="tmp", channel_coords=list(range(NUM_DATAPOINTS_PER_YEAR)))
    param_dataset = create_reference_raster(filepath=params_path, channel_name="layer", channel_coords=["SOS", "EOS", "sNDVI", "wNDVI"])

    for i, path in enumerate(filepaths):
        with torch.no_grad():

            try:
                minicube = xr.open_dataset(path, engine="h5netcdf")
            except OSError:
                continue

            missing_dates = check_missing_timestamps(minicube, START_YEAR, END_YEAR)
            if missing_dates:
                minicube = minicube.reindex(
                    time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
                )

            try:
                ndvi = minicube.s2_ndvi.where((minicube.s2_mask == 0) & minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7])).values
                forest_mask = (minicube.FOREST_MASK.values > 0.8)
            except AttributeError:
                continue

            lon_left, lon_right = minicube.lon[0].values, minicube.lon[-1].values
            lat_bottom, lat_top = minicube.lat[-1].values, minicube.lat[0].values

            # convert to torch
            ndvi = torch.from_numpy(ndvi).to(device)
            doy = torch.from_numpy(get_doy(minicube.time.values)).to(device)
            forest_mask = torch.from_numpy(forest_mask).to(device)

            # mask out non-forest pixels
            ndvi[:, ~forest_mask] = torch.nan

            # get test
            ndvi_test = ndvi[SPLIT_IDX:]
            doy_test = doy[SPLIT_IDX:]

            # features for network
            dem = torch.from_numpy(minicube.DEM.values).to(device)
            fc = torch.from_numpy(minicube.FC.values).to(device)
            fh = torch.from_numpy(minicube.FH.values).to(device)
            slope = torch.from_numpy(minicube.slope.values).to(device)
            easting = torch.from_numpy(minicube.easting.values).to(device)
            northing = torch.from_numpy(minicube.northing.values).to(device)
            twi = torch.from_numpy(minicube.twi.values).to(device)
            rugg = torch.from_numpy(minicube.rugg.values).to(device)
            curv = torch.from_numpy(minicube.curv.values).to(device)

            pressure = torch.from_numpy(minicube.era5_sp.values[:SPLIT_IDX]).to(device)
            annual_pressure = torch.nanmean(pressure.view(END_YEAR - START_YEAR, 73, H, W), axis=0)
            press_mean = torch.mean(annual_pressure, axis=0)
            press_std = torch.std(annual_pressure, axis=0)

            temperature = torch.from_numpy(minicube.era5_t2m.values[:SPLIT_IDX]).to(device)
            annual_temperature = torch.nanmean(temperature.view(END_YEAR - START_YEAR, 73, H, W), axis=0)
            temp_mean = torch.mean(annual_temperature, axis=0)
            temp_std = torch.std(annual_temperature, axis=0)

            precipitation = torch.from_numpy(minicube.era5_tp.values[:SPLIT_IDX]).to(device)
            annual_precipitation = torch.nanmean(precipitation.view(END_YEAR - START_YEAR, 73, H, W), axis=0)
            precip_mean = torch.mean(annual_precipitation, axis=0)
            precip_std = torch.std(annual_precipitation, axis=0)

            fc[fc == -9999] = torch.nan
            inp = torch.stack([dem, fc, fh, slope, easting, northing, twi, rugg, curv, press_mean, press_std, temp_mean, temp_std, precip_mean, precip_std], axis=0)  # 11 x H x W
            inp = (inp - means) / stds
            inp = torch.nan_to_num(inp, nan=0.0)

            # only forward forest pixels
            masked_inp = inp[:, forest_mask]
            masked_inp = masked_inp.permute(1, 0)  # B x 11

            preds = encoder(masked_inp.float())
            paramsl = preds[:, [0, 1, 2, 3, 4, 5]]
            paramsu = torch.cat([preds[:, [0, 1, 2, 3]], preds[:, [4, 5]] + nn.functional.softplus(preds[:, [6, 7]])], axis=1)

            paramsl = rectify_parameters(paramsl)
            paramsu = rectify_parameters(paramsu)

            # taking the mean is only necessary for M and m, because the time parameters are shared
            params = (paramsl + paramsu) / 2
            params = convert_params(params)

            params_map = torch.full((H, W, 4), torch.nan, device=device)
            params_map[forest_mask, :] = params

            project_patch(params_path, lon_left, lon_right, lat_bottom, lat_top, params_map.cpu().numpy(), forest_mask.cpu().numpy(), param_dataset, "layer")

            # save the anomaly score
            t_test = doy_test * T_SCALE
            ndvi_lower_pred = double_logistic_function(t_test, paramsl).permute(1, 0)
            ndvi_upper_pred = double_logistic_function(t_test, paramsu).permute(1, 0)

            res_ndvi_lower_pred = torch.full(ndvi_test.shape, torch.nan, device=device)
            res_ndvi_upper_pred = torch.full(ndvi_test.shape, torch.nan, device=device)

            res_ndvi_lower_pred[:, forest_mask] = ndvi_lower_pred
            res_ndvi_upper_pred[:, forest_mask] = ndvi_upper_pred

            iqr = res_ndvi_upper_pred - res_ndvi_lower_pred

            anomaly_score = torch.full(ndvi_test.shape, torch.nan, device=device, dtype=float)
            anomaly_score[ndvi_test > res_ndvi_upper_pred] = ((ndvi_test - res_ndvi_upper_pred) / iqr)[ndvi_test > res_ndvi_upper_pred]
            anomaly_score[ndvi_test < res_ndvi_lower_pred] = ((ndvi_test - res_ndvi_lower_pred) / iqr)[ndvi_test < res_ndvi_lower_pred]
            anomaly_score[(ndvi_test >= res_ndvi_lower_pred) & (ndvi_test <= res_ndvi_upper_pred)] = 0.0

            project_patch(anoms_path, lon_left, lon_right, lat_bottom, lat_top, anomaly_score.cpu().numpy(), forest_mask.cpu().numpy(), anom_dataset, "tmp")

        if (i + 1) % 100 == 0:
            print("done {}".format(i + 1))

    consolidate(anoms_path, anom_dataset, channel_name="time", channel_coords=get_dates(), post_processing=True)
    consolidate(params_path, param_dataset, channel_name="layer", channel_coords=["SOS", "EOS", "sNDVI", "wNDVI"], post_processing=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('DIMPEO Inference')
    parser.add_argument('-n', '--name', type=str, default="dimpeo_inference")
    parser.add_argument('--encoder-name', type=str)
    parser.add_argument('--features', type=str, default=None)
    args = parser.parse_args()

    inference(args.name, args.encoder_name, args.features)