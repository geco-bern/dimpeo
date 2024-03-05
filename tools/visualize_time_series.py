# %%
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.patchwise_dataset import PatchwiseDataset

path = "/Volumes/Macintosh HD/Users/davidbruggemann/OneDrive - epfl.ch/DIMPEO/data/tmp3_train.h5"

ds = PatchwiseDataset(path, pixelwise=True)

dataloader = DataLoader(
    ds,
    batch_size=128,
    shuffle=True,
)
dataloader_iter = iter(dataloader)


# %%
plot_idx = 0
# get random samples
dd = next(dataloader_iter)
dst = dd['spatiotemporal']
fig, ax = plt.subplots(2, 6, figsize=(15, 6), sharey=True)
# plotting NDVI
ax[0, 0].plot(dst[0, :, plot_idx])
ax[0, 1].plot(dst[1, :, plot_idx])
ax[0, 2].plot(dst[2, :, plot_idx])
ax[0, 3].plot(dst[3, :, plot_idx])
ax[0, 4].plot(dst[4, :, plot_idx])
ax[0, 5].plot(dst[5, :, plot_idx])
ax[1, 0].plot(dst[6, :, plot_idx])
ax[1, 1].plot(dst[7, :, plot_idx])
ax[1, 2].plot(dst[8, :, plot_idx])
ax[1, 3].plot(dst[9, :, plot_idx])
ax[1, 4].plot(dst[10, :, plot_idx])
ax[1, 5].plot(dst[11, :, plot_idx])
ax[0, 0].set_ylim([0, 1])
ax[1, 0].set_ylim([0, 1])
ax[0, 0].set_ylabel("NDVI")
ax[1, 0].set_ylabel("NDVI")


# %%
