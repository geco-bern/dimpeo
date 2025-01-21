import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import d2_pinball_score

from neural_network.mlp import MLP


YEARS_IN_TRAIN = 6  # first six years in train, last year in test
SPLIT_IDX = 73 * YEARS_IN_TRAIN
T_SCALE = 1.0 / 365.0  # rescale target

# rescale input (stats obtained from full dataset)
MEANS = {
    "lon": 8.34458,
    "lat": 46.32019,
    "dem": 1104.5027,
    "fc": 43.102547,
    "fh": 15.638395,
    "slope": 26.418936,
    "easting": -0.0069493465,
    "northing": 0.062305786,
    "twi": 2.8649843,
    "rugg": 8.363732,
    "curv": 6.1888146e-05,
    "press_mean": 87901.86,
    "press_std": 293.46072,
    "temp_mean": 280.14288,
    "temp_std": 6.8551936,
    "precip_mean": 0.019856384,
    "precip_std": 0.009813356,
}
STDS = {
    "lon": 0.9852665,
    "lat": 0.61929077,
    "dem": 409.8894,
    "fc": 40.840992,
    "fh": 30.395235,
    "slope": 13.358144,
    "easting": 0.69276434,
    "northing": 0.7184338,
    "twi": 1.861385,
    "rugg": 5.4581103,
    "curv": 0.0061930786,
    "press_mean": 5674.5396,
    "press_std": 52.65978,
    "temp_mean": 3.3523095,
    "temp_std": 0.4047956,
    "precip_mean": 0.0034253746,
    "precip_std": 0.0024573584,
}


class H5Dataset():

    all_features = ["lon", "lat", "dem", "fc", "fh", "slope", "easting", "northing", "twi", "rugg", "curv", "press_mean", "press_std", "temp_mean", "temp_std", "precip_mean", "precip_std"]

    def __init__(self, file_path, features=None):
        self.file_path = file_path
        self.features = features if features is not None else self.all_features
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file.get("doy"))
            self.ndvi_ds = file.get("ndvi")[:]
            self.doy_ds = file.get("doy")[:].astype(int)
            if "lon" in self.features or "lat" in self.features:
                self.lon_lat_ds = file.get("lon_lat")[:]
            if "dem" in self.features:
                self.dem_ds = file.get("dem")[:]
            if "fc" in self.features:
                self.fc_ds = file.get("fc")[:]
                self.fc_ds[self.fc_ds == -9999] = np.nan
            if "fh" in self.features:
                self.fh_ds = file.get("fh")[:]
            if "slope" in self.features:
                self.slope_ds = file.get("slope")[:]
            if "easting" in self.features:
                self.easting_ds = file.get("easting")[:]
            if "northing" in self.features:
                self.northing_ds = file.get("northing")[:]
            if "twi" in self.features:
                self.twi_ds = file.get("twi")[:]
            if "rugg" in self.features:
                self.rugg_ds = file.get("rugg")[:]
            if "curv" in self.features:
                self.curv_ds = file.get("curv")[:]
            if "press_mean" in self.features:
                self.press_mean_ds = file.get("press_mean")[:]
            if "press_std" in self.features:
                self.press_std_ds = file.get("press_std")[:]
            if "temp_mean" in self.features:
                self.temp_mean_ds = file.get("temp_mean")[:]
            if "temp_std" in self.features:
                self.temp_std_ds = file.get("temp_std")[:]
            if "precip_mean" in self.features:
                self.precip_mean_ds = file.get("precip_mean")[:]
            if "precip_std" in self.features:
                self.precip_std_ds = file.get("precip_std")[:]

        self.missingness = np.load(os.path.join(os.path.dirname(file_path), "missingness.npy"))

    def __getitem__(self, index):
        ndvi = self.ndvi_ds[index]
        doy = self.doy_ds[index]

        inp = []
        for f in self.features:
            if f == "lon":
                inp.append([self.lon_lat_ds[index, 0]])
            elif f == "lat":
                inp.append([self.lon_lat_ds[index, 1]])
            elif f == "dem":
                inp.append(self.dem_ds[index])
            elif f == "fc":
                inp.append(self.fc_ds[index])
            elif f == "fh":
                inp.append(self.fh_ds[index])
            elif f == "slope":
                inp.append(self.slope_ds[index])
            elif f == "easting":
                inp.append(self.easting_ds[index])
            elif f == "northing":
                inp.append(self.northing_ds[index])
            elif f == "twi":
                inp.append(self.twi_ds[index])
            elif f == "rugg":
                inp.append(self.rugg_ds[index])
            elif f == "curv":
                inp.append(self.curv_ds[index])
            elif f == "press_mean":
                inp.append(self.press_mean_ds[index])
            elif f == "press_std":
                inp.append(self.press_std_ds[index])
            elif f == "temp_mean":
                inp.append(self.temp_mean_ds[index])
            elif f == "temp_std":
                inp.append(self.temp_std_ds[index])
            elif f == "precip_mean":
                inp.append(self.precip_mean_ds[index])
            elif f == "precip_std":
                inp.append(self.precip_std_ds[index])
        inp = np.concatenate(inp)
        return ndvi, doy, inp

    def __len__(self):
        return self.dataset_len


def double_logistic_function(t, params):
    sos, mat_minus_sos, sen, eos_minus_sen, M, m = torch.split(params, 1, dim=1)
    mat_minus_sos = nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = nn.functional.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-10))
    sigmoid_sen_eos = nn.functional.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-10))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def objective_pinball(params, t, ndvi, nan_mask, alpha=0.5, weights=None):
    ndvi_pred = double_logistic_function(t, params)
    diff = ndvi - ndvi_pred
    loss = torch.max(torch.mul(alpha, diff), torch.mul((alpha - 1), diff))
    # we need to reweight the quantiles to prevent a degenerate solution
    if weights is not None:
        loss = loss * weights.repeat(YEARS_IN_TRAIN).unsqueeze(0)
    return torch.mean(loss[~nan_mask])


def train(name, data_path, features=None):
    torch.manual_seed(1)

    writer = SummaryWriter(os.path.join(os.environ["SAVE_DIR"], f"runs/{name}"))

    max_iterations = 500000
    lr = 0.001
    lr_decay_rate = 0.01
    batch_size = 256
    device = 'cuda'

    print("Starting model with:")
    print("name = {}".format(name))

    print("Loading dataset into RAM...")
    if isinstance(features, str):
        features = features.split(",")
    ds = H5Dataset(data_path, features)
    # load avg missingness as a function of DOY (73 entries in total)
    missingness = torch.from_numpy(ds.missingness).to(device)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=32)

    print("Using features: {}".format(ds.features))

    means_pt = torch.tensor([MEANS[f] for f in ds.features]).to(device).unsqueeze(0)
    stds_pt = torch.tensor([STDS[f] for f in ds.features]).to(device).unsqueeze(0)

    # this model has ~470k parameters
    encoder = MLP(d_in=len(ds.features), d_out=8, n_blocks=8, d_block=256, dropout=0, skip_connection=True).to(device)
    # we don't care about overfitting --> weight decay not strictly necessary
    bias_params = [p for name, p in encoder.named_parameters() if 'bias' in name]
    others = [p for name, p in encoder.named_parameters() if 'bias' not in name]
    optimizer = torch.optim.AdamW([
                {'params': others},
                {'params': bias_params, 'weight_decay': 0}
            ], weight_decay=1e-4, lr=lr)

    # in total we have 20314711 pixels, with 6 parameters per pixel
    # so technically the data has ~120 Mio degrees of freedom
    print("Number of parameters: {}".format(sum(p.numel() for p in encoder.parameters() if p.requires_grad)))
    print("Starting training...")

    n_iterations = 0
    n_epochs = 0
    stop = False
    while True:
        print("Starting epoch {}".format(n_epochs + 1))
        
        for sample in loader:
            ndvi, doy, inp = sample

            # remove implausible values
            outlier_mask = (ndvi > 1) | (ndvi < -0.1)

            nan_mask = torch.isnan(ndvi)
            nan_mask = nan_mask | outlier_mask

            # separate into train/test
            ndvi_train = ndvi[:, :SPLIT_IDX]
            doy_train = doy[:, :SPLIT_IDX]
            nan_mask_train = nan_mask[:, :SPLIT_IDX]

            inp = inp.float().to(device)

            # standardize input
            inp = (inp - means_pt) / stds_pt

            # impute possible missing values with mean (should be the case only for fc)
            inp = torch.nan_to_num(inp, nan=0.0)

            t_ndvi_train = ndvi_train.float().to(device)
            t_doy_train = doy_train.float().to(device) * T_SCALE
            t_nan_mask_train = nan_mask_train.to(device)

            preds = encoder(inp.float()) # B x 8
            paramsl = preds[:, [0, 1, 2, 3, 4, 5]]  # B x 6
            paramsu = torch.cat([preds[:, [0, 1, 2, 3]], preds[:, [4, 5]] + nn.functional.softplus(preds[:, [6, 7]])], axis=1)

            lossl = objective_pinball(paramsl, t_doy_train, t_ndvi_train, t_nan_mask_train, alpha=0.25, weights=missingness)
            lossu = objective_pinball(paramsu, t_doy_train, t_ndvi_train, t_nan_mask_train, alpha=0.75, weights=missingness)
            loss = lossl + lossu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_lrate = lr * (lr_decay_rate ** (n_iterations / max_iterations))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if (n_iterations + 1) % 500 == 0:
                writer.add_scalar("Loss/train", loss, n_iterations)
                for pi, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar("LearningRate[{}]".format(pi), param_group['lr'], n_iterations)

            if (n_iterations + 1) % 1000 == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey=True)

                    t_fit = torch.linspace(0, 365, 1000).unsqueeze(0).repeat(paramsl.shape[0], 1)
                    ndvi_lower = double_logistic_function(t_fit * T_SCALE, paramsl.cpu())
                    ndvi_upper = double_logistic_function(t_fit * T_SCALE, paramsu.cpu())

                    random_indices = np.random.choice(np.arange(batch_size), size=4, replace=False)
                    for pl_idx, bi in enumerate(random_indices):
                        row, col = divmod(pl_idx, 2)
                        masked_ndvi_train = ndvi_train[bi][~nan_mask_train[bi]]
                        masked_doy_train = doy_train[bi][~nan_mask_train[bi]]
                        ax[row, col].scatter(masked_doy_train, masked_ndvi_train, label='Observed NDVI')
                        ax[row, col].fill_between(t_fit[bi], ndvi_lower[bi], ndvi_upper[bi], alpha=0.2, color='red')
                        ax[0, 0].set_ylabel("NDVI")
                        ax[1, 0].set_ylabel("NDVI")

                    writer.add_figure("Fit", fig, n_iterations)

                    all_ndvi_lower = double_logistic_function(doy_train * T_SCALE, paramsl.cpu())
                    all_ndvi_upper = double_logistic_function(doy_train * T_SCALE, paramsu.cpu())
                    all_masked_ndvi_lower = all_ndvi_lower[~nan_mask_train]
                    all_masked_ndvi_upper = all_ndvi_upper[~nan_mask_train]
                    all_masked_ndvi_train = ndvi_train[~nan_mask_train]
                    d2_score_lower = d2_pinball_score(all_masked_ndvi_train, all_masked_ndvi_lower, alpha=0.25)
                    d2_score_upper = d2_pinball_score(all_masked_ndvi_train, all_masked_ndvi_upper, alpha=0.75)

                    writer.add_scalar("D2PinballScoreLower/train", d2_score_lower, n_iterations)
                    writer.add_scalar("D2PinballScoreUpper/train", d2_score_upper, n_iterations)

            if (n_iterations + 1) % 10000 == 0:
                torch.save(encoder.state_dict(), os.path.join(os.environ["SAVE_DIR"], f"encoder_{name}.pt"))

            n_iterations += 1
            if n_iterations >= max_iterations:
                stop = True
                break

        if stop:
            break

        n_epochs += 1
        writer.add_scalar("Epochs", n_epochs, n_iterations)

    torch.save(encoder.state_dict(), os.path.join(os.environ["SAVE_DIR"], f"encoder_{name}.pt"))

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DIMPEO Training')
    parser.add_argument('-n', '--name', type=str, default="dimpeo_training")
    parser.add_argument('--data-path', type=str, default=os.path.join(os.environ["PROC_DIR"], "nn_dataset.h5"))
    parser.add_argument('--features', type=str)
    args = parser.parse_args()

    train(args.name, args.data_path, args.features)
