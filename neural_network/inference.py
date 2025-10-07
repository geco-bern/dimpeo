import argparse
import numpy as np
import glob
import torch
import torch.nn as nn
import xarray as xr
import os
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

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


def chunk_iterator(zarr_array, chunk_size):
    n = zarr_array.shape[0]
    for i in range(0, n, chunk_size):
        yield slice(i, min(i + chunk_size, n))


def inference(encoder_path, features, name):

    batch_size = 8192

    data_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"

    out_zarr = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"

    ds = ZarrDataset(data_path)
    # loader = DataLoader(
    #     ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    device = "cuda"

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

    num_columns = [column for feat_name in ds.num_features for column in ds.mapping_features[feat_name]]
    species_columns = ds.mapping_features["tree_species"]
    habitat_columns = ds.mapping_features["habitat"]

    
    N = ds.dataset_len
    print(f"Number of samples: {N}")

    root = zarr.open_group(out_zarr, mode='a')
    if "params" in root:
        del root["params"]
    feat_grp = root.require_group('params')

    feat_grp.create_array(
        name="params_lower",
        shape=(N, 6),
        chunks=(4000, 6),
        dtype="float32",
    )

    feat_grp.create_array(
        name="params_upper",
        shape=(N, 6),
        chunks=(4000, 6),
        dtype="float32",
    )

    description = (
        "Double logistic function parameters:\n"
        "- sos: start of season (day of year)\n"
        "- mat_minus_sos: duration of green-up\n"
        "- sen: start of senescence\n"
        "- eos_minus_sen: duration of senescence\n"
        "- M: maximum NDVI\n"
        "- m: minimum NDVI"
    )

    feat_grp["params_lower"].attrs["description"] = "Lower bound parameters of the seasonal NDVI cycle.\n" + description
    feat_grp["params_upper"].attrs["description"] = "Upper bound parameters of the seasonal NDVI cycle.\n" + description
    feat_grp.attrs["encoder_path"] = encoder_path

    with torch.no_grad():
        for slc in tqdm(chunk_iterator(ds.feat_array, 4000), total=(N + 4000 - 1) // 4000):
            feat = torch.from_numpy(ds.feat_array[slc, :]).to(device).float()

            feat = feat.to(device)
            feat_num = feat[:, num_columns]
            feat_species = feat[:, species_columns].int()
            feat_habitat = feat[:, habitat_columns]

            feat_num[feat_num == -9999] = 0
            feat_species[feat_species == 255] = 17

            # standardize input
            feat_num = (feat_num - means_pt) / stds_pt

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

            paramsl = rectify_parameters(paramsl)
            paramsu = rectify_parameters(paramsu)

            feat_grp['params_lower'][slc, :] = paramsl.detach().cpu().numpy()
            feat_grp['params_upper'][slc, :] = paramsu.detach().cpu().numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("DIMPEO Inference")
    parser.add_argument("-n", "--name", type=str, default="dimpeo_inference")
    parser.add_argument("--encoder-path", type=str)
    parser.add_argument(
        "--features",
        type=str,
        default="dem,slope,easting,northing,twi,tri,mean_curv,profile_curv,plan_curv,roughness,median_forest_height,forest_mix_rate,tree_species,habitat",
    )
    args = parser.parse_args()

    print("Running inference...")
    print("Using encoder:", args.encoder_path)
    print("Using features:", args.features)

    inference(
        args.encoder_path,
        args.features,
        args.name,
    )