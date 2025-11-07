import torch
import zarr
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

class ShuffleDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        zarr_store = zarr.open(file_path, mode="r", zarr_format=3)
        self.feat_array = zarr_store['merged_features']

        self.ndvi = zarr_store['ndvi']
        self.ndsi = zarr_store['ndsi']
        self.dataset_len = self.ndvi.shape[0]
        self.timesteps = self.ndvi.shape[1]
        self.n_features = self.feat_array.shape[1]

    def __getitem__(self, idx):
        ndvi = self.ndvi[idx]
        ndsi = self.ndsi[idx]
        features = self.feat_array[idx]
        return ndvi, ndsi, features

    def __len__(self):
        return self.dataset_len

def write_shuffled_copy(dataloader, target_zarr, n_samples, n_timesteps, n_features, chunk_rows=8192):
    root = zarr.open_group(target_zarr, mode="w", zarr_format=3)
    ndvi_out = root.create_array(
        "ndvi",
        shape=(n_samples, n_timesteps),
        chunks=(chunk_rows, n_timesteps),
        dtype=np.int16,
        compressors=[],
    )
    feat_out = root.create_array(
        "merged_features",
        shape=(n_samples, n_features),
        chunks=(chunk_rows, n_features),
        dtype=dataloader.dataset.feat_array.dtype,
        compressors=[],
    )

    offset = 0
    for batch in tqdm(dataloader, desc="Writing shuffled NDVI", total=len(dataloader)):
        ndvi, ndsi, feat = batch
        ndvi = ndvi.numpy().astype(np.int16)
        ndsi = ndsi.numpy().astype(np.int16)
        feat = feat.numpy()
        mask = (ndsi > 4300) & (ndsi < 10000)
        ndvi[mask] = -2**15
        ndvi_out[offset:offset + ndvi.shape[0], :] = ndvi
        feat_out[offset:offset + feat.shape[0], :] = feat
        offset += ndvi.shape[0]

batch_size = 512

ds = ShuffleDataset("/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr")
loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

write_shuffled_copy(loader,
    "/data_2/scratch/sbiegel/processed/ndvi_dataset_filtered_shuffled.zarr",
    n_samples=len(ds),
    n_timesteps=ds.timesteps,
    n_features=ds.n_features,
    chunk_rows=batch_size
)