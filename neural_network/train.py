import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from sklearn.metrics import d2_pinball_score
from tqdm import tqdm
from datetime import datetime

from neural_network.mlp import MLPWithEmbeddings
from neural_network.helpers import get_split_indices
from neural_network.dataset import ZarrDataset, MEANS, STDS


# YEARS_IN_TRAIN = [2017, 2018, 2019, 2020, 2021, 2022]
# YEARS_IN_TEST = [2023]
T_SCALE = 1.0 / 365.0  # rescale target


def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = nn.functional.sigmoid(
        -2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10)
    )
    sigmoid_sen_eos = nn.functional.sigmoid(
        -2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10)
    )
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def objective_pinball(params, t, ndvi, nan_mask, alpha=0.5, weights=None):
    ndvi_pred = double_logistic_function(t, params)
    diff = ndvi - ndvi_pred
    loss = torch.max(torch.mul(alpha, diff), torch.mul((alpha - 1), diff))
    # we need to reweight the quantiles to prevent a degenerate solution
    if weights is not None:
        loss = loss * weights.unsqueeze(0)
    return torch.mean(loss[~nan_mask])


def train(name, data_path, features=None):
    torch.manual_seed(1)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"/data_2/scratch/sbiegel/processed/runs/{name}/{run_id}")
    checkpoint_dir = f"/data_2/scratch/sbiegel/processed/checkpoints/{name}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    max_iterations = 500000
    max_batches_per_epoch = 10000
    lr = 0.001
    lr_decay_rate = 0.01
    batch_size = 512
    device = "cuda"

    print("Starting model with:")
    print("name = {}".format(name))

    print("Loading dataset...")
    if isinstance(features, str):
        features = features.split(",")
    ds = ZarrDataset(data_path, features)
    missingness = ds.missingness
    missingness = torch.from_numpy(missingness).to(device)

    sampler = RandomSampler(ds, replacement=True, num_samples=batch_size * max_batches_per_epoch)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Using features: {}".format(ds.features))

    means_pt = torch.tensor([MEANS[f] for f in ds.num_features]).unsqueeze(0)
    stds_pt = torch.tensor([STDS[f] for f in ds.num_features]).unsqueeze(0)

    nr_num_features = ds.nr_num_features
    nr_species = ds.nr_tree_species
    nr_habitats = ds.nr_habitats

    # this model has ~475k parameters
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

    bias_params = [p for name, p in encoder.named_parameters() if "bias" in name]
    others = [p for name, p in encoder.named_parameters() if "bias" not in name]
    optimizer = torch.optim.AdamW(
        [{"params": others}, {"params": bias_params, "weight_decay": 0}],
        weight_decay=1e-4,
        lr=lr,
    )

    print(
        "Number of parameters: {}".format(
            sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        )
    )
    print("Starting training...")

    n_iterations = 0
    n_epochs = 0
    stop = False
    while True:
        print("Starting epoch {}".format(n_epochs + 1))

        for sample in tqdm(loader):
            ndvi, ndsi, feat = sample

            nan_mask = torch.isnan(ndvi) | (ndvi == -2**15) | (ndvi == 2**15 - 1)
            ndvi = ndvi.float() / 10000.0
            ndsi = ndsi.float() / 10000.0
            snow_mask = (ndsi > 0.43) & (ndsi < 1.0)
            outlier_mask = (ndvi > 1) | (ndvi < -0.1)
            nan_mask = nan_mask | outlier_mask | snow_mask

            t = torch.from_numpy(ds.t).to(device, non_blocking=True)

            feat_num = feat[:, ds.num_feature_indices]
            feat_species = feat[:, ds.mapping_features["tree_species"]].int()
            feat_habitat = feat[:, ds.mapping_features["habitat"]].int()
            feat_species[feat_species == 255] = 16

            # standardize input
            feat_num = (feat_num - means_pt) / stds_pt

            feat_num = feat_num.to(device, non_blocking=True)
            feat_species = feat_species.to(device, non_blocking=True)
            feat_habitat = feat_habitat.to(device, non_blocking=True)

            t_ndvi_train = ndvi.float().to(device, non_blocking=True)
            t_nan_mask_train = nan_mask.to(device, non_blocking=True)

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

            lossl = objective_pinball(
                paramsl,
                    t,
                t_ndvi_train,
                t_nan_mask_train,
                alpha=0.25,
                weights=missingness,
            )
            lossu = objective_pinball(
                paramsu,
                    t,
                t_ndvi_train,
                t_nan_mask_train,
                alpha=0.75,
                weights=missingness,
            )

            # Add constraint to ensure periodicity
                t_start = torch.full((feat.shape[0], 1), 0, device=device)
                t_end   = torch.full((feat.shape[0], 1), 1, device=device)
            startl = double_logistic_function(t_start, paramsl)
            endl = double_logistic_function(t_end, paramsl)
            periodic_loss_l = torch.mean((startl - endl) ** 2)

            startu = double_logistic_function(t_start, paramsu)
            endu = double_logistic_function(t_end, paramsu)
            periodic_loss_u = torch.mean((startu - endu) ** 2)

            lambda_periodic = 0.1

            loss = lossl + lossu + lambda_periodic * (periodic_loss_l + periodic_loss_u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_lrate = lr * (lr_decay_rate ** (n_iterations / max_iterations))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

            if (n_iterations + 1) % 500 == 0:
                writer.add_scalar("Loss/train", loss, n_iterations)
                for pi, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(
                        "LearningRate[{}]".format(pi), param_group["lr"], n_iterations
                    )

            if (n_iterations + 1) % 500 == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey=True)

                    t_fit = (
                        torch.linspace(0, 365, 1000)
                        .unsqueeze(0)
                        .repeat(paramsl.shape[0], 1)
                    )
                    ndvi_lower = double_logistic_function(
                        t_fit * T_SCALE, paramsl.cpu()
                    )
                    ndvi_upper = double_logistic_function(
                        t_fit * T_SCALE, paramsu.cpu()
                    )

                    random_indices = np.random.choice(
                        np.arange(batch_size), size=4, replace=False
                    )
                    for pl_idx, bi in enumerate(random_indices):
                        row, col = divmod(pl_idx, 2)
                        masked_ndvi_train = ndvi[bi][~nan_mask[bi]]
                        doy_expanded = doy.unsqueeze(0).expand(nan_mask.shape[0], -1).cpu()
                        masked_doy_train = doy_expanded[bi][~nan_mask[bi]].cpu()
                        ax[row, col].scatter(
                            masked_doy_train, masked_ndvi_train, label="Observed NDVI"
                        )
                        ax[row, col].fill_between(
                            t_fit[bi],
                            ndvi_lower[bi],
                            ndvi_upper[bi],
                            alpha=0.2,
                            color="red",
                        )
                        ax[0, 0].set_ylabel("NDVI")
                        ax[1, 0].set_ylabel("NDVI")

                    writer.add_figure(f"Fit/iter_{n_iterations+1}", fig, n_iterations)

                    all_ndvi_lower = double_logistic_function(
                        doy.cpu() * T_SCALE, paramsl.cpu()
                    )
                    all_ndvi_upper = double_logistic_function(
                        doy.cpu() * T_SCALE, paramsu.cpu()
                    )
                    all_masked_ndvi_lower = all_ndvi_lower[~nan_mask]
                    all_masked_ndvi_upper = all_ndvi_upper[~nan_mask]
                    all_masked_ndvi_train = ndvi[~nan_mask]
                    d2_score_lower = d2_pinball_score(
                        all_masked_ndvi_train, all_masked_ndvi_lower, alpha=0.25
                    )
                    d2_score_upper = d2_pinball_score(
                        all_masked_ndvi_train, all_masked_ndvi_upper, alpha=0.75
                    )

                    writer.add_scalar(
                        "D2PinballScoreLower/train", d2_score_lower, n_iterations
                    )
                    writer.add_scalar(
                        "D2PinballScoreUpper/train", d2_score_upper, n_iterations
                    )

            if (n_iterations + 1) % 10000 == 0:
                torch.save(encoder.state_dict(), f"{checkpoint_dir}/encoder_iter{n_iterations+1}.pt")

            n_iterations += 1
            if n_iterations >= max_iterations:
                stop = True
                break

        if stop:
            break

        n_epochs += 1
        writer.add_scalar("Epochs", n_epochs, n_iterations)

    torch.save(
        encoder.state_dict(),
        f"{checkpoint_dir}/encoder.pt"
    )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DIMPEO Training")
    parser.add_argument("-n", "--name", type=str, default="dimpeo_training")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="dem,slope,easting,northing,twi,tri,mean_curv,profile_curv,plan_curv,roughness,median_forest_height,forest_mix_rate,tree_species,habitat",
    )
    args = parser.parse_args()

    train(args.name, args.data_path, args.features)
