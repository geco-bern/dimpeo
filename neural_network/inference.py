import argparse
import numpy as np
import glob
import torch
import torch.nn as nn
import xarray as xr
import os
import zarr
from torch.utils.data import DataLoader

from neural_network.mlp import MLPWithEmbeddings
from neural_network.train import (
    double_logistic_function,
    MEANS,
    STDS,
    T_SCALE,
)
from neural_network.helpers import (
    get_doy,
    check_missing_timestamps,
    create_reference_raster,
    project_patch,
    consolidate,
    get_dates,
    convert_params,
    get_split_indices,
    NUM_DATAPOINTS_PER_YEAR,
    START_YEAR,
    END_YEAR,
    H,
    W,
)
from neural_network.dataset import ZarrDataset


def rectify_parameters(params):
    # this is necessary because the model is not identifiable (several parameter choices yield the same NDVI curve)
    # params: sos, mat_minus_sos, sen, eos_minus_sen, M, m
    inverted_mask = params[:, 0] > params[:, 2]

    rec_params = params.clone()
    rec_params[inverted_mask, 0] = params[inverted_mask, 2]
    rec_params[inverted_mask, 1] = params[inverted_mask, 3]
    rec_params[inverted_mask, 2] = params[inverted_mask, 0]
    rec_params[inverted_mask, 3] = params[inverted_mask, 1]
    rec_params[inverted_mask, 4] = (
        params[inverted_mask, 5] + params[inverted_mask, 5] - params[inverted_mask, 4]
    )
    # rec_params[inverted_mask, 5] = params[inverted_mask, 5]

    return rec_params


def inference(encoder_path, features, name):

    batch_size = 256

    data_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset.zarr"

    out_zarr = "/data_2/scratch/sbiegel/processed/sample_seasonal_cycle_parameter_preds.zarr"

    ds = ZarrDataset(data_path)
    loader = DataLoader(
        ds,
        batch_size=100,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    device = "cuda"

    # N = ds.dataset_len

    # output_ds = zarr.create_array(
    #     name="param",
    #     store=out_zarr,
    #     shape=(100, 6),
    #     chunks=(4000,),
    #     zarr_format=3,
    #     dtype=
    # )

    if features is None:
        features = [
            "dem",
            "slope",
            "easting",
            "northing",
            "twi",
            "tri",
            "mean_curv",
            "profile_curv",
            "plan_curv",
            "roughness",
            "median_forest_height",
            "tree_species",
            "habitat",
        ]
    elif isinstance(features, str):
        features = features.split(",")

    # train_indices = get_split_indices(YEARS_IN_TRAIN)
    # test_indices = get_split_indices(YEARS_IN_TEST)

    nr_num_features = ds.nr_num_features
    nr_species = ds.nr_tree_species
    nr_habitats = ds.nr_habitats

    encoder = MLPWithEmbeddings(
        d_num=nr_num_features,
        d_out=8,
        n_blocks=8,
        d_block=256,
        dropout=0.0,
        skip_connection=True,
        n_species=nr_species,
        species_emb_dim=4,
        n_habitats=nr_habitats,
        habitat_emb_dim=8,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder.eval()

    means_pt = torch.tensor([MEANS[f] for f in ds.num_features]).to(device).unsqueeze(0)
    stds_pt = torch.tensor([STDS[f] for f in ds.num_features]).to(device).unsqueeze(0)

    for sample in loader:
        ndvi, feat = sample

        nan_mask = (ndvi == -2**15) | (ndvi == 2**15 - 1)
        ndvi[nan_mask] = np.nan
        # ndvi = ndvi.float() / 10000.0

        feat = feat.to(device)
        feat_num = feat[:, [column for feat_name in ds.num_features for column in ds.mapping_features[feat_name]]]
        feat_species = feat[:, ds.mapping_features["tree_species"]].int()
        feat_habitat = feat[:, ds.mapping_features["habitat"]]

        # standardize input
        feat_num = (feat_num - means_pt) / stds_pt

        feat_num[feat_num == -9999] = 0
        feat_species[feat_species == 255] = 17

        preds = encoder(
                feat_num,
                feat_species,
                feat_habitat,
            )
        paramsl = preds[:, [0, 1, 2, 3, 4, 5]]  # B x 6
        paramsu = torch.cat(
            [
                preds[:, [0, 1, 2, 3]],
                preds[:, [4, 5]] + nn.functional.softplus(preds[:, [6, 7]]),
            ],
            axis=1,
        )

        doy = ds.doy

        param_names = ["sos", "mat_minus_sos", "sen", "eos_minus_sen", "M", "m"]
        description = (
            "Double logistic function parameters:\n"
            "- sos: start of season (day of year)\n"
            "- mat_minus_sos: duration of green-up\n"
            "- sen: start of senescence\n"
            "- eos_minus_sen: duration of senescence\n"
            "- M: maximum NDVI\n"
            "- m: minimum NDVI"
        )

        ds = xr.Dataset(
            data_vars={
                "ndvi": (("pixel", "time"), ndvi.numpy()),
                "doy": ("time", doy),
                "params_lower": (("pixel", "param"), paramsl.detach().cpu().numpy()),
                "params_upper": (("pixel", "param"), paramsu.detach().cpu().numpy()),
                "dates": ("dates", ds.dates)
            },
            coords={
                "param": param_names
            },
            attrs={
                "title": "NDVI seasonal cycle and double logistic parameters",
                "source": encoder_path,
                "note": "Each row corresponds to a forest pixel"
            }
        )

        ds["params_lower"].attrs["description"] = "Lower bound parameters of the seasonal NDVI cycle.\n" + description
        ds["params_upper"].attrs["description"] = "Upper bound parameters of the seasonal NDVI cycle.\n" + description
        ds["ndvi"].attrs["units"] = "unitless (vegetation index)"
        ds["doy"].attrs["description"] = "Day of year for each time step"

        ds.to_zarr(out_zarr, mode='a')

        break





    # anom_dataset = create_reference_raster(
    #     filepath=anoms_path,
    #     channel_name="tmp",
    #     channel_coords=list(range(NUM_DATAPOINTS_PER_YEAR * len(YEARS_IN_TEST))),
    #     metadata={
    #         "negative_anomaly_id": 0,
    #         "normal_id": 1,
    #         "positive_anomaly_id": 2,
    #         "missing_id": 255,
    #     },
    # )
    # param_dataset = create_reference_raster(
    #     filepath=params_path,
    #     channel_name="layer",
    #     channel_coords=["SOS", "EOS", "sNDVI", "wNDVI"],
    # )

    # if save_npy:
    #     params_list, anoms_list, coords_list, forest_list, ndvi_list, ndvi_raw_list, cloud_list, scl_list = [], [], [], [], [], [], [], []

#     for i, path in enumerate(filepaths):
#         with torch.no_grad():

#             try:
#                 minicube = xr.open_dataset(path, engine="h5netcdf")
#             except OSError:
#                 continue

#             missing_dates = check_missing_timestamps(minicube)
#             if missing_dates:
#                 minicube = minicube.reindex(
#                     time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
#                 )

#             try:
#                 ndvi = minicube.s2_ndvi.where(
#                     (minicube.s2_mask == 0) & minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7])
#                 ).values
#                 forest_mask = minicube.FOREST_MASK.values > 0.8
#                 ndvi_raw = minicube.s2_ndvi.values
#                 cloud_mask = minicube.s2_mask.values
#                 scl = minicube.s2_SCL.values
#             except AttributeError:
#                 continue

#             lon_left, lon_right = minicube.lon[0].values, minicube.lon[-1].values
#             lat_bottom, lat_top = minicube.lat[-1].values, minicube.lat[0].values

#             # convert to torch
#             ndvi = torch.from_numpy(ndvi).to(device)
#             ndvi_raw = torch.from_numpy(ndvi_raw).to(device)
#             cloud_mask = torch.from_numpy(cloud_mask).to(device)
#             scl = torch.from_numpy(scl).to(device)
#             doy = torch.from_numpy(get_doy(minicube.time.values)).to(device)
#             forest_mask = torch.from_numpy(forest_mask).to(device)

#             # mask out non-forest pixels
#             ndvi[:, ~forest_mask] = torch.nan
#             ndvi_raw[:, ~forest_mask] = torch.nan

#             # get test
#             ndvi_test = ndvi[test_indices]
#             ndvi_raw_test = ndvi_raw[test_indices]
#             cloud_mask_test = cloud_mask[test_indices]
#             scl_test = scl[test_indices]
#             doy_test = doy[test_indices]

#             # features for network
#             dem = torch.from_numpy(minicube.DEM.values).to(device)
#             fc = torch.from_numpy(minicube.FC.values).to(device)
#             fh = torch.from_numpy(minicube.FH.values).to(device)
#             slope = torch.from_numpy(minicube.slope.values).to(device)
#             easting = torch.from_numpy(minicube.easting.values).to(device)
#             northing = torch.from_numpy(minicube.northing.values).to(device)
#             twi = torch.from_numpy(minicube.twi.values).to(device)
#             rugg = torch.from_numpy(minicube.rugg.values).to(device)
#             curv = torch.from_numpy(minicube.curv.values).to(device)

#             pressure = torch.from_numpy(minicube.era5_sp.values[train_indices]).to(
#                 device
#             )
#             annual_pressure = torch.nanmean(
#                 pressure.view(END_YEAR - START_YEAR, NUM_DATAPOINTS_PER_YEAR, H, W),
#                 axis=0,
#             )
#             press_mean = torch.mean(annual_pressure, axis=0)
#             press_std = torch.std(annual_pressure, axis=0)

#             temperature = torch.from_numpy(minicube.era5_t2m.values[train_indices]).to(
#                 device
#             )
#             annual_temperature = torch.nanmean(
#                 temperature.view(END_YEAR - START_YEAR, NUM_DATAPOINTS_PER_YEAR, H, W),
#                 axis=0,
#             )
#             temp_mean = torch.mean(annual_temperature, axis=0)
#             temp_std = torch.std(annual_temperature, axis=0)

#             precipitation = torch.from_numpy(minicube.era5_tp.values[train_indices]).to(
#                 device
#             )
#             annual_precipitation = torch.nanmean(
#                 precipitation.view(
#                     END_YEAR - START_YEAR, NUM_DATAPOINTS_PER_YEAR, H, W
#                 ),
#                 axis=0,
#             )
#             precip_mean = torch.mean(annual_precipitation, axis=0)
#             precip_std = torch.std(annual_precipitation, axis=0)

#             fc[fc == -9999] = torch.nan
#             inp = torch.stack(
#                 [
#                     dem,
#                     fc,
#                     fh,
#                     slope,
#                     easting,
#                     northing,
#                     twi,
#                     rugg,
#                     curv,
#                     press_mean,
#                     press_std,
#                     temp_mean,
#                     temp_std,
#                     precip_mean,
#                     precip_std,
#                 ],
#                 axis=0,
#             )  # 11 x H x W
#             inp = (inp - means) / stds
#             inp = torch.nan_to_num(inp, nan=0.0)

#             # only forward forest pixels
#             masked_inp = inp[:, forest_mask]
#             masked_inp = masked_inp.permute(1, 0)  # B x 11

#             preds = encoder(masked_inp.float())
#             paramsl = preds[:, [0, 1, 2, 3, 4, 5]]
#             paramsu = torch.cat(
#                 [
#                     preds[:, [0, 1, 2, 3]],
#                     preds[:, [4, 5]] + nn.functional.softplus(preds[:, [6, 7]]),
#                 ],
#                 axis=1,
#             )

#             paramsl = rectify_parameters(paramsl)
#             paramsu = rectify_parameters(paramsu)

#             # taking the mean is only necessary for M and m, because the time parameters are shared
#             params = (paramsl + paramsu) / 2

#             if save_npy:
#                 params_map_save = torch.full((H, W, 6), torch.nan, device=device)
#                 params_map_save[forest_mask, :] = params
#                 params_list.append(params_map_save.cpu().numpy())

#             params = convert_params(params)

#             params_map = torch.full((4, H, W), torch.nan, device=device)
#             params_map[:, forest_mask] = params.permute(1, 0)

#             project_patch(
#                 params_path,
#                 lon_left,
#                 lon_right,
#                 lat_bottom,
#                 lat_top,
#                 params_map.cpu().numpy(),
#                 forest_mask.cpu().numpy(),
#                 param_dataset,
#                 channel_name="layer",
#                 nx=W,
#                 ny=H,
#             )

#             # save the anomaly score
#             t_test = doy_test * T_SCALE
#             ndvi_lower_pred = double_logistic_function(t_test, paramsl).permute(1, 0)
#             ndvi_upper_pred = double_logistic_function(t_test, paramsu).permute(1, 0)

#             res_ndvi_lower_pred = torch.full(ndvi_test.shape, torch.nan, device=device)
#             res_ndvi_upper_pred = torch.full(ndvi_test.shape, torch.nan, device=device)

#             res_ndvi_lower_pred[:, forest_mask] = ndvi_lower_pred
#             res_ndvi_upper_pred[:, forest_mask] = ndvi_upper_pred

#             iqr = res_ndvi_upper_pred - res_ndvi_lower_pred

#             anomaly_score = torch.full(
#                 ndvi_test.shape, torch.nan, device=device, dtype=float
#             )
#             anomaly_score[ndvi_test > res_ndvi_upper_pred] = (
#                 (ndvi_test - res_ndvi_upper_pred) / iqr
#             )[ndvi_test > res_ndvi_upper_pred]
#             anomaly_score[ndvi_test < res_ndvi_lower_pred] = (
#                 (ndvi_test - res_ndvi_lower_pred) / iqr
#             )[ndvi_test < res_ndvi_lower_pred]
#             anomaly_score[
#                 (ndvi_test >= res_ndvi_lower_pred) & (ndvi_test <= res_ndvi_upper_pred)
#             ] = 0.0

#             if save_npy:
#                 anoms_list.append(anomaly_score.cpu().numpy())
#                 coords_list.append(np.array([lon_left, lon_right, lat_bottom, lat_top]))
#                 forest_list.append(forest_mask.cpu().numpy())
#                 ndvi_list.append(ndvi_test.cpu().numpy())
#                 ndvi_raw_list.append(ndvi_raw_test.cpu().numpy())
#                 cloud_list.append(cloud_mask_test.cpu().numpy())
#                 scl_list.append(scl_test.cpu().numpy())

#             project_patch(
#                 anoms_path,
#                 lon_left,
#                 lon_right,
#                 lat_bottom,
#                 lat_top,
#                 anomaly_score.cpu().numpy(),
#                 forest_mask.cpu().numpy(),
#                 anom_dataset,
#                 channel_name="tmp",
#                 nx=W,
#                 ny=H,
#             )

#         if (i + 1) % 100 == 0:
#             print("done {}".format(i + 1))

#     if save_npy:
#         params_list = np.stack(params_list, axis=0)
#         anoms_list = np.stack(anoms_list, axis=0)
#         coords_list = np.stack(coords_list, axis=0)
#         forest_list = np.stack(forest_list, axis=0)
#         ndvi_list = np.stack(ndvi_list, axis=0)
#         ndvi_raw_list = np.stack(ndvi_raw_list, axis=0)
#         cloud_list = np.stack(cloud_list, axis=0)
#         scl_list = np.stack(scl_list, axis=0)

#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"parameters_{name}.npy"), params_list
#         )
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"anomalies_{name}.npy"), anoms_list
#         )
#         np.save(os.path.join(os.environ["SAVE_DIR"], f"coords_{name}.npy"), coords_list)
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"forest_mask_{name}.npy"), forest_list
#         )
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"ndvi_{name}.npy"), ndvi_list
#         )
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"ndvi_raw_{name}.npy"), ndvi_raw_list
#         )
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"cloud_mask_{name}.npy"), cloud_list
#         )
#         np.save(
#             os.path.join(os.environ["SAVE_DIR"], f"scl_{name}.npy"), scl_list
#         )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("DIMPEO Inference")
    parser.add_argument("-n", "--name", type=str, default="dimpeo_inference")
    parser.add_argument("--encoder-path", type=str)
    parser.add_argument(
        "--features",
        type=str,
        default="dem,slope,easting,northing,twi,tri,mean_curv,profile_curv,plan_curv,roughness,median_forest_height,tree_species,habitat",
    )
    args = parser.parse_args()

    inference(
        args.encoder_path,
        args.features,
        args.name,
    )

#     consolidate(
#         anoms_path,
#         channel_name="time",
#         channel_coords=get_dates(YEARS_IN_TEST),
#         discretize=True,
#     )
#     consolidate(
#         params_path,
#         channel_name="layer",
#         channel_coords=["SOS", "EOS", "sNDVI", "wNDVI"],
#     )
