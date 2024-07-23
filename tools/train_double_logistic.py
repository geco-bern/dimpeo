import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import d2_pinball_score

from model.mlp import MLP


PATH = "/data_1/scratch_1/processed/nn_dataset.h5"
YEARS_IN_TRAIN = 6  # first six years in train, last year in test
SPLIT_IDX = 73 * YEARS_IN_TRAIN


class H5Dataset():
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file.get("doy"))
            self.ndvi_ds = file.get("ndvi")[:]
            self.doy_ds = file.get("doy")[:].astype(int)
            self.lon_lat_ds = file.get("lon_lat")[:]
            self.dem_ds = file.get("dem")[:]
            self.fc_ds = file.get("fc")[:]
            self.fc_ds[self.fc_ds == -9999] = np.nan
            self.fh_ds = file.get("fh")[:]
            self.slope_ds = file.get("slope")[:]
            self.easting_ds = file.get("easting")[:]
            self.northing_ds = file.get("northing")[:]
            self.twi_ds = file.get("twi")[:]

        self.missingness = np.load(os.path.join(os.path.dirname(file_path), "missingness.npy"))

    def __getitem__(self, index):
        ndvi = self.ndvi_ds[index]
        doy = self.doy_ds[index]
        lon_lat = self.lon_lat_ds[index]
        dem = self.dem_ds[index]
        fc = self.fc_ds[index]
        fh = self.fh_ds[index]
        slope = self.slope_ds[index]
        easting = self.easting_ds[index]
        northing = self.northing_ds[index]
        twi = self.twi_ds[index]

        inp = np.concatenate([lon_lat, dem, fc, fh, slope, easting, northing, twi])

        return ndvi, doy, inp

    def __len__(self):
        return self.dataset_len


def double_logistic_function(t, params):
    M, m, sos, mat_minus_sos, sen, eos_minus_sen = torch.split(params, 1, dim=1)
    mat_minus_sos = nn.functional.softplus(mat_minus_sos)
    eos_minus_sen = nn.functional.softplus(eos_minus_sen)
    sigmoid_sos_mat = nn.functional.sigmoid(-2 * (2 * sos + mat_minus_sos - 2 * t) / (mat_minus_sos + 1e-8))
    sigmoid_sen_eos = nn.functional.sigmoid(-2 * (2 * sen + eos_minus_sen - 2 * t) / (eos_minus_sen + 1e-8))
    return (M - m) * (sigmoid_sos_mat - sigmoid_sen_eos) + m


def objective_pinball(params, t, ndvi, nan_mask, alpha=0.5, weights=None):
    ndvi_pred = double_logistic_function(t, params)
    diff = ndvi - ndvi_pred
    loss = torch.max(torch.mul(alpha, diff), torch.mul((alpha - 1), diff))
    # we need to reweight the qunatiles to prevent a degenerate solution
    if weights is not None:
        loss = loss * weights.repeat(YEARS_IN_TRAIN).unsqueeze(0)
    fil_losses = loss[~nan_mask]
    if torch.any(torch.isnan(fil_losses)):
        print("NaN in loss !")
        print(fil_losses)
        raise ValueError
    return torch.mean(fil_losses)


def train():
    torch.manual_seed(1)

    writer = SummaryWriter()

    max_iterations = 250000
    lr = 0.001
    lr_decay_rate = 0.1
    batch_size = 256
    device = 'cuda'

    # rescale target
    NDVI_SCALE = 1.0
    T_SCALE = 1.0 / 365.0

    print("Loading dataset into RAM...")
    ds = H5Dataset(PATH)
    # load avg missingness as a function of DOY (73 entries in total)
    missingness = torch.from_numpy(ds.missingness).to(device)
    # TODO: profiling to determine num_workers
    loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=32)

    # rescale input (stats obtained from full dataset)
    MEANS = torch.tensor([
        8.34458,  # lon
        46.32019,  # lat
        1104.5027,  # dem
        43.102547,  # fc
        15.638395,  # fh
        26.418936,  # slope
        -0.0069493465,  # easting
        0.062305786,  # northing
        2.8649843,  # twi
        ]).to(device).unsqueeze(0)
    STDS = torch.tensor([
        0.9852665,  # lon
        0.61929077,  # lat
        409.8894,  # dem
        40.840992,  # fc
        30.395235,  # fh
        13.358144,  # slope
        0.69276434,  # easting
        0.7184338,  # northing
        1.861385,  # twi
        ]).to(device).unsqueeze(0)

    # this model has ~470k parameters
    encoder = MLP(d_in=9, d_out=12, n_blocks=8, d_block=256, dropout=0, skip_connection=True).to(device)
    # we don't care about overfitting --> no weight decay
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

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
            ndvi_train, ndvi_test = ndvi[:, :SPLIT_IDX], ndvi[:, SPLIT_IDX:]
            doy_train, doy_test = doy[:, :SPLIT_IDX], doy[:, SPLIT_IDX:]
            nan_mask_train, nan_mask_test = nan_mask[:, :SPLIT_IDX], nan_mask[:, SPLIT_IDX:]

            # TODO: add positional encoding

            inp = inp.float().to(device)  # B x 9
            # standardize input
            inp = (inp - MEANS) / STDS

            # impute possible missing values with mean (should be the case only for fc)
            inp = torch.nan_to_num(inp, nan=0.0)

            t_ndvi_train = ndvi_train.float().to(device) * NDVI_SCALE
            t_doy_train = doy_train.float().to(device) * T_SCALE
            t_nan_mask_train = nan_mask_train.to(device)

            # we predict a residual
            preds = encoder(inp.float()) # B x 12
            paramsl = preds[:, :6]  # B x 6
            try:
                lossl = objective_pinball(paramsl, t_doy_train, t_ndvi_train, t_nan_mask_train, alpha=0.25, weights=missingness)
            except ValueError as e:
                print("Error in iteration {}".format(n_iterations))
                raise e

            paramsu = preds[:, 6:]
            try:
                lossu = objective_pinball(paramsu, t_doy_train, t_ndvi_train, t_nan_mask_train, alpha=0.75, weights=missingness)
            except ValueError as e:
                print("Error in iteration {}".format(n_iterations))
                raise e
            
            loss = lossl + lossu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update lr
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
                    ndvi_lower = double_logistic_function(t_fit * T_SCALE, paramsl.cpu()) / NDVI_SCALE
                    ndvi_upper = double_logistic_function(t_fit * T_SCALE, paramsu.cpu()) / NDVI_SCALE

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

                    # TODO: computing scoring rule on val/test
                    all_ndvi_lower = double_logistic_function(doy_train * T_SCALE, paramsl.cpu()) / NDVI_SCALE
                    all_ndvi_upper = double_logistic_function(doy_train * T_SCALE, paramsu.cpu()) / NDVI_SCALE
                    all_masked_ndvi_lower = all_ndvi_lower[~nan_mask_train]
                    all_masked_ndvi_upper = all_ndvi_upper[~nan_mask_train]
                    all_masked_ndvi_train = ndvi_train[~nan_mask_train]
                    d2_score_lower = d2_pinball_score(all_masked_ndvi_train, all_masked_ndvi_lower, alpha=0.25)
                    d2_score_upper = d2_pinball_score(all_masked_ndvi_train, all_masked_ndvi_upper, alpha=0.75)

                    writer.add_scalar("D2PinballScoreLower/train", d2_score_lower, n_iterations)
                    writer.add_scalar("D2PinballScoreUpper/train", d2_score_upper, n_iterations)

            if (n_iterations + 1) % 10000 == 0:
                torch.save(encoder.state_dict(), "/data_1/scratch_1/dbrueggemann/nn/encoder_full.pt")

            n_iterations += 1

            if n_iterations >= max_iterations:
                stop = True
                break

        if stop:
            break

        n_epochs += 1
        writer.add_scalar("Epochs", n_epochs, n_iterations)

    torch.save(encoder.state_dict(), "/data_1/scratch_1/dbrueggemann/nn/encoder_full.pt")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()