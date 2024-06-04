import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from data.patchwise_dataset import PatchwiseDataset
from density_estimation.kde_npphen import get_density_kde_npphen
from density_estimation.kde_rkde import get_density_kde_rkde
from density_estimation.kde_sklearn import get_density_kde_sklearn
from density_estimation.kde_statsmodels import get_density_kde_statsmodels


def fit_kde(file_path, kde_type="npphen"):
    ds = PatchwiseDataset(
        file_path,
        pixelwise=False,
        annual=False,
        spatiotemporal_features=["s2_ndvi"],
        spatial_features=["drought_mask", "valid_mask"],
    )

    torch.manual_seed(1)
    loader = DataLoader(ds, batch_size=1, drop_last=True, shuffle=True)
    loader_iter = iter(loader)

    use_context = False

    while True:

        try:
            sample = next(loader_iter)
        except StopIteration:
            # reinitialize data loader
            loader_iter = iter(loader)
            sample = next(loader_iter)

        if use_context:
            # sample one pixel and its window
            h, w = sample["spatiotemporal"].shape[2:4]
            i = 0
            while i < 100:
                rh = torch.randint(low=1, high=h - 1, size=(1,)).item()
                rw = torch.randint(low=1, high=w - 1, size=(1,)).item()
                is_valid = sample["spatial"][0, rh - 1 : rh + 2, rw - 1 : rw + 2, 1]
                if np.all(np.array(is_valid, dtype=bool)):
                    break
                i += 1
            else:
                print("couldn't find valid patch")
                continue
            ndvi = (
                (
                    sample["spatiotemporal"][0, :, rh - 1 : rh + 2, rw - 1 : rw + 2, 0]
                    + 1
                )
                * 5000
            ).to(
                int
            )  # T
            ano_ndvi, ref_ndvi = ndvi[:73], ndvi[73:]
            ano_ndvi = ano_ndvi[:, 1, 1]  # center pixel
            ref_ndvi = np.reshape(ref_ndvi, -1)

            dgs = sample["dgs"][0].to(int)  # T
            dgs[dgs == 366] = 365  # to account for gap years
            ano_dgs, ref_dgs = dgs[:73], dgs[73:]
            ref_dgs = np.repeat(ref_dgs, 9)

            ano = np.stack((ano_dgs, ano_ndvi), axis=1)
            ref = np.stack((ref_dgs, ref_ndvi), axis=1)

        else:
            # sample one pixel
            h, w = sample["spatiotemporal"].shape[2:4]
            while True:
                rh = torch.randint(h, size=(1,)).item()
                rw = torch.randint(w, size=(1,)).item()
                is_valid = sample["spatial"][0, rh, rw, 1]
                if is_valid:
                    break
            ndvi = ((sample["spatiotemporal"][0, :, rh, rw, 0] + 1) * 5000).to(int)  # T
            dgs = sample["dgs"][0].to(int)  # T
            dgs[dgs == 366] = 365  # to account for gap years
            data = np.stack((dgs, ndvi), axis=1)

            # take first year as test year
            ano, ref = data[:73], data[73:]
            # ano, ref = data[-73:], data[:-73]

        ref = ref[~np.isnan(ref).any(axis=1)]  # remove rows with nan
        ano = ano[~np.isnan(ano).any(axis=1)]  # remove rows with nan

        start_t = time.time()

        if kde_type == "npphen":
            kernel_density = get_density_kde_npphen(ref)
        elif kde_type == "statsmodels":
            kernel_density = get_density_kde_statsmodels(ref)
        elif kde_type == "sklearn":
            kernel_density = get_density_kde_sklearn(ref)
        elif kde_type == "rkde":
            kernel_density = get_density_kde_rkde(ref, type_rho="huber", periodic=True)
        else:
            raise ValueError

        end_t = time.time()
        print("KDE took {} s".format(end_t - start_t))

        kernel_density = normalize(
            kernel_density, "l1", axis=1
        )  # normalize at each DGS
        kernel_density /= np.sum(kernel_density)

        def get_reference_frequency_distribution(pdf):
            flat_pdf = pdf.flatten()
            sort_vals_idx = np.argsort(flat_pdf)[::-1]
            sort_rfd = np.cumsum(flat_pdf[sort_vals_idx])
            return sort_rfd[np.argsort(sort_vals_idx)].reshape(pdf.shape)

        def nan_argmax(a):
            valid_max_idx = (
                np.count_nonzero(
                    np.where(
                        a == np.max(a, axis=1, keepdims=True),
                        1,
                        0,
                    ),
                    axis=1,
                )
                == 1
            )
            max_idx = np.argmax(a, axis=1).astype(float)
            max_idx[~valid_max_idx] = np.nan
            return max_idx

        rfd = get_reference_frequency_distribution(kernel_density)

        max_density = nan_argmax(kernel_density)

        row_anom = np.abs(
            np.linspace(0, 10000, num=500)[np.newaxis, :] - ano[:, 1][:, np.newaxis]
        )
        row_anom2 = np.argmin(row_anom, axis=1)
        anom_rfd = rfd[ano[:, 0] - 1, row_anom2]
        anom_rfd_perc = np.round(100 * anom_rfd)

        xx, yy = np.mgrid[0:365, 0:500]

        fig, ax = plt.subplots(
            2, 1, figsize=(6, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        levels = [0, 0.5, 0.75, 0.9, 0.95]
        cfset = ax[0].contourf(xx, yy, rfd, levels=levels, cmap="hot", alpha=0.6)
        cset = ax[0].contour(xx, yy, rfd, levels=levels, colors="grey", linewidths=1)
        ax[0].clabel(cset, inline=1, fontsize=10)
        ax[0].plot(range(len(max_density)), max_density, c="darkred")
        # ax[0].set_ylim(0, 500)
        # ax[0].set_yticks([0, 100, 200, 300, 400, 500])
        # ax[0].set_yticklabels([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
        ax[0].set_ylim(300, 500)
        ax[0].set_yticks([300, 350, 400, 450, 500])
        ax[0].set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[0].set_ylabel("NDVI")
        cbar = fig.colorbar(cfset)
        cbar.set_label("RFD")

        # plot the ano points
        ax[0].scatter(ano[:, 0] - 1, row_anom2, marker="o", c="k", s=2)

        threshold = 90
        colors = np.array(["grey"] * len(anom_rfd_perc))
        colors[anom_rfd_perc > threshold] = "red"
        ax[1].bar(ano[:, 0] - 1, anom_rfd_perc, color=colors)
        ax[1].axhline(y=threshold, color="k", linestyle="-")
        ax[1].set_ylabel("RFD [%]")
        ax[1].set_ylim(0, 100)

        ax[1].set_xlim(0, 364)
        ax[1].set_xlabel("DGS")

        pos = ax[0].get_position()
        pos2 = ax[1].get_position()
        ax[1].set_position([pos.x0, pos2.y0, pos.width, pos2.height])

        fig.suptitle("has_drought: {}".format(sample["spatial"][0, rh, rw, 0].item()))

        plt.show(block=False)
        input("Press key to continue...")
        plt.close()


if __name__ == "__main__":
    path = "/Volumes/Macintosh HD/Users/davidbruggemann/OneDrive/DIMPEO/data/tmp7_train.h5"
    kde_type = "rkde"
    fit_kde(path, kde_type=kde_type)
