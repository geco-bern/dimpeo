# %%
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.patchwise_dataset import PatchwiseDataset

path = "/Volumes/Macintosh HD/Users/davidbruggemann/OneDrive - epfl.ch/DIMPEO/data/tmp2_train.h5"

ds = PatchwiseDataset(path, pixelwise=True)

dataloader = DataLoader(
    ds,
    batch_size=128,
    shuffle=True,
)
dataloader_iter = iter(dataloader)


# %%
plot_idx = 4
# get random samples
d = next(dataloader_iter)
fig, ax = plt.subplots(2, 3, sharey=True)
# plotting NDVI
ax[0, 0].plot(d[0, plot_idx, :])
ax[0, 1].plot(d[1, plot_idx, :])
ax[0, 2].plot(d[2, plot_idx, :])
ax[1, 0].plot(d[3, plot_idx, :])
ax[1, 1].plot(d[4, plot_idx, :])
ax[1, 2].plot(d[5, plot_idx, :])
ax[0, 0].set_ylim([0, 1])
ax[1, 0].set_ylim([0, 1])
ax[0, 0].set_ylabel("NDVI")
ax[1, 0].set_ylabel("NDVI")


# %%
