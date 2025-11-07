import argparse
import torch
import zarr
from tqdm import tqdm
import numpy as np

from neural_network.mlp import MLPWithEmbeddings
from neural_network.train import (
    MEANS,
    STDS,
    double_logistic_function,
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

    return rec_params


def chunk_iterator(zarr_array, chunk_size):
    n = zarr_array.shape[0]
    for i in range(0, n, chunk_size):
        yield slice(i, min(i + chunk_size, n))


def inference(encoder_path, features, name):

    data_path = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"

    ds = ZarrDataset(data_path)

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
            "forest_mix_rate",
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
        d_out=18,
        n_blocks=8,
        d_block=256,
        dropout=0.0,
        skip_connection=True,
        n_species=nr_species,
        species_emb_dim=4,
        n_habitats=nr_habitats,
        habitat_emb_dim=8,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    means_pt = torch.tensor([MEANS[f] for f in ds.num_features]).to(device).unsqueeze(0)
    stds_pt = torch.tensor([STDS[f] for f in ds.num_features]).to(device).unsqueeze(0)

    num_columns = [column for feat_name in ds.num_features for column in ds.mapping_features[feat_name]]
    species_columns = ds.mapping_features["tree_species"]
    habitat_columns = ds.mapping_features["habitat"]
    
    N = ds.dataset_len
    T = ds.timesteps
    print(f"Number of samples: {N}")

    root_params = zarr.open_group(data_path, mode='a')
    # if "params" in root:
    #     del root["params"]
    feat_grp = root_params.create_group('params_2')

    feat_grp.create_array(
        name="params_lower",
        shape=(N, 6),
        chunks=(4000, 6),
        dtype="float32",
    )

    feat_grp.create_array(
        name="params_median",
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
    feat_grp["params_median"].attrs["description"] = "Median parameters of the seasonal NDVI cycle.\n" + description
    feat_grp["params_upper"].attrs["description"] = "Upper bound parameters of the seasonal NDVI cycle.\n" + description
    feat_grp.attrs["encoder_path"] = encoder_path

    feat_grp_preds = root_params.create_group('ndvi_preds')
    feat_grp_preds.create_array(
        name="ndvi_pred_lower",
        shape=(N, T),
        chunks=(4000, T),
        dtype="int16",
    )
    feat_grp_preds.create_array(
        name="ndvi_pred_median",
        shape=(N, T),
        chunks=(4000, T),
        dtype="int16",
    )
    feat_grp_preds.create_array(
        name="ndvi_pred_upper",
        shape=(N, T),
        chunks=(4000, T),
        dtype="int16",
    )

    root_params.create_array(
        name="anomalies",
        shape=(N, T),
        chunks=(4000, T),
        dtype="int8",
        fill_value=127,
        compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle),
    )
    root_params['anomalies'].attrs["description"] = (
        "NDVI anomaly values (int8) for each pixel in the reference grid. 0 means no anomaly, "
        "1 means positive anomaly, -1 means negative anomaly. Missing values (no coverage, no forest) are indicated by 127."
        "Masked values (clouds, shadows, snow, outliers) are indicated by -128."
    )
    root_params['anomalies'].attrs["encoder_path"] = encoder_path
    ndvi_anomaly_arr = root_params['anomalies']

    root_params.create_array(
        name="anomaly_scores",
        shape=(N, T),
        chunks=(4000, T),
        dtype="float32",
        fill_value=np.nan,
        compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle),
    )
    root_params['anomaly_scores'].attrs["encoder_path"] = encoder_path
    ndvi_anomaly_score_arr = root_params['anomaly_scores']

    t = torch.tensor(ds.t).to(device).float()

    with torch.inference_mode():
        for i, slc in enumerate(tqdm(chunk_iterator(ds.feat_array, 4000), total=(N + 4000 - 1) // 4000)):
            feat = torch.from_numpy(ds.feat_array[slc, :]).to(device).float()
            ndvi = torch.from_numpy(ds.ndvi[slc, :]).to(device).float()
            ndsi = torch.from_numpy(ds.ndsi[slc, :]).to(device).float()

            feat_num = feat[:, num_columns]
            feat_species = feat[:, species_columns].int()
            feat_habitat = feat[:, habitat_columns].int()

            feat_species[feat_species == 255] = 16

            # standardize input
            feat_num = (feat_num - means_pt) / stds_pt

            preds = encoder(
                    feat_num,
                    feat_species,
                    feat_habitat,
                )
            
            paramsl = preds[:, [0, 1, 2, 3, 4, 5]]  # B x 6
            paramsm = preds[:, [6, 7, 8, 9, 10, 11]]  # B x 6
            paramsu = preds[:, [12, 13, 14, 15, 16, 17]]  # B x 6

            paramsl = rectify_parameters(paramsl)
            paramsm = rectify_parameters(paramsm)
            paramsu = rectify_parameters(paramsu)

            feat_grp['params_lower'][slc, :] = paramsl.detach().cpu().numpy()
            feat_grp['params_median'][slc, :] = paramsm.detach().cpu().numpy()
            feat_grp['params_upper'][slc, :] = paramsu.detach().cpu().numpy()

            ndvi_lower = double_logistic_function(t, paramsl).detach().cpu().numpy()
            ndvi_median = double_logistic_function(t, paramsm).detach().cpu().numpy()
            ndvi_upper = double_logistic_function(t, paramsu).detach().cpu().numpy()

            ndvi = ndvi.detach().cpu().numpy()
            ndsi = ndsi.detach().cpu().numpy()

             # Identify data availability
            is_unavailable = ndvi == 2**15 - 1
            is_masked = ndvi == -2**15
            is_outlier = ((ndvi > 10000) | (ndvi < -1000)) & ~is_unavailable & ~is_masked
            is_nan = np.isnan(ndvi)
            is_snow = (ndsi >= 4300) & (ndsi <= 10000)

            valid_mask = ~(is_unavailable | is_masked | is_outlier | is_snow | is_nan)

            ndvi = ndvi / 10000.0

            feat_grp_preds['ndvi_pred_lower'][slc, :] = (ndvi_lower * 10000.0).round().astype(np.int16)
            feat_grp_preds['ndvi_pred_median'][slc, :] = (ndvi_median * 10000.0).round().astype(np.int16)
            feat_grp_preds['ndvi_pred_upper'][slc, :] = (ndvi_upper * 10000.0).round().astype(np.int16)

            iqr = ndvi_upper - ndvi_lower

            lower_thresh = (ndvi_lower - 1.5 * iqr)
            upper_thresh = (ndvi_upper + 1.5 * iqr)

            # Compute anomalies only for valid NDVI
            is_lower_anomaly = (ndvi < lower_thresh) & valid_mask
            is_upper_anomaly = (ndvi > upper_thresh) & valid_mask

            # Initialize output
            anomalies = np.zeros_like(ndvi, dtype=np.int8)
            anomalies[is_lower_anomaly] = -1
            anomalies[is_upper_anomaly] = 1
            anomalies[is_masked | is_outlier | is_nan | is_snow] = -128
            anomalies[is_unavailable] = 127

            ndvi_anomaly_arr[slc, :] = anomalies

            score = np.zeros_like(ndvi, dtype=np.float32)

            below = (ndvi < ndvi_lower) & valid_mask
            above = (ndvi > ndvi_upper) & valid_mask

            score[below] = -(ndvi_lower[below] - ndvi[below]) / iqr[below]
            score[above] = (ndvi[above] - ndvi_upper[above]) / iqr[above]
            score[is_masked | is_outlier | is_nan | is_snow] = np.nan
            score[is_unavailable] = np.nan

            ndvi_anomaly_score_arr[slc, :] = score

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("DIMPEO Inference")
    parser.add_argument("-n", "--name", type=str, default="dimpeo_inference")
    parser.add_argument("--encoder-path", type=str, default="/data_2/scratch/sbiegel/processed/checkpoints/dimpeo_training/logistic/20251016_163631/encoder.pt")
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