import zarr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset

# rescale input (stats obtained from full dataset)
MEANS = {
    "dem": 1071.1402587890625,
    "slope": 26.603004455566406,
    "easting": 0.01555223111063242,
    "northing": -0.06615273654460907,
    "twi": 2.7378883361816406,
    "mean_curv": 0.00045366896665655077,
    "profile_curv": 0.00016974707250483334,
    "plan_curv": 0.002614102093502879,
    "tri": 2.885868787765503,
    "roughness": 2.9208521842956543,
    "median_forest_height": 24.493475,
    "forest_mix_rate": 0.2507593148494662,
}
STDS = {
    "dem": 450.6628112792969,
    "slope": 14.156021118164062,
    "easting": 0.6557915806770325,
    "northing": 0.6876032948493958,
    "twi": 1.7116827964782715,
    "mean_curv": 0.005201074760407209,
    "profile_curv": 0.005349245388060808,
    "plan_curv": 0.0478176586329937,
    "tri": 2.302125930786133,
    "roughness": 2.2183449268341064,
    "median_forest_height": 6.8817625,
    "forest_mix_rate": 0.6688715032184841,
}


class ChunkedZarrDataset(IterableDataset):
    def __init__(
        self,
        file_path,
        features=None,
        batch_size=512,
        chunk_size=8192,
        shuffle_chunks=True,
        seed=42,
    ):
        super().__init__()
        self.file_path = file_path
        self.features = features
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.base_seed = seed

        store = zarr.open(file_path, mode="r", zarr_format=3)
        original_store = zarr.open("/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr", mode="r", zarr_format=3)
        self.ndvi = store["ndvi"]
        self.feat = store["merged_features"]
        self.mapping_features = original_store["merged_features"].attrs["feature_columns"]
        self.missingness = original_store["missingness"][:]
        self.dates = np.array(
            [pd.to_datetime(d.decode("utf-8")) for d in original_store["dates"][:]]
        )
        dtindex = pd.DatetimeIndex(self.dates)
        self.doy = dtindex.dayofyear.to_numpy()
        is_leap = dtindex.is_leap_year.astype(int)
        self.t = (self.doy - 1) / (365 + is_leap)

        self.dataset_len = self.ndvi.shape[0]
        self.timesteps = self.ndvi.shape[1]
        self.n_chunks = int(np.ceil(self.dataset_len / self.chunk_size))
        self.num_features = [
            f for f in self.features if f not in ['tree_species', 'habitat']
        ]

        # total number of minibatches per epoch
        self.n_batches = int((self.dataset_len + self.batch_size - 1) // self.batch_size)
        self.nr_num_features = len(self.num_features)
        self.num_feature_indices = [i for f in self.num_features for i in self.mapping_features[f]]
        self.nr_tree_species = 17
        self.nr_habitats = 46

    def set_epoch(self, epoch):
        # allows reproducible per-epoch reshuffle
        self.epoch = epoch
        self.rng = np.random.default_rng(self.base_seed + epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if hasattr(self, "epoch"):
            epoch = self.epoch
        else:
            epoch = 0

        if worker_info is not None:
            wid = worker_info.id
            nworkers = worker_info.num_workers
        else:
            wid = 0
            nworkers = 1
        seed = self.base_seed + wid + epoch * 1000
        self.rng = np.random.default_rng(seed)

        store = zarr.open(self.file_path, mode="r", zarr_format=3)
        self.ndvi = store["ndvi"]
        self.feat = store["merged_features"]

        chunk_indices = np.arange(self.n_chunks)
        if self.shuffle_chunks:
            self.rng.shuffle(chunk_indices)

        chunk_indices = chunk_indices[wid::nworkers]

        for cid in chunk_indices:
            start = cid * self.chunk_size
            stop = min((cid + 1) * self.chunk_size, self.dataset_len)

            ndvi_chunk = np.asarray(self.ndvi[start:stop])
            feat_chunk = np.asarray(self.feat[start:stop])

            # local shuffle in memory
            order = self.rng.permutation(len(ndvi_chunk))
            ndvi_chunk = ndvi_chunk[order]
            feat_chunk = feat_chunk[order]

            for i in range(0, len(ndvi_chunk), self.batch_size):
                j = min(i + self.batch_size, len(ndvi_chunk))
                yield (
                    torch.from_numpy(ndvi_chunk[i:j]).float(),
                    torch.from_numpy(feat_chunk[i:j]).float(),
                )

    def __len__(self):
        """Return total number of minibatches per epoch (across all chunks).
        """
        return self.n_batches


class ZarrDataset:

    all_features = [
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

    def __init__(self, file_path, features=None):
        self.file_path = file_path
        self.features = features if features is not None else self.all_features
        zarr_store = zarr.open(file_path, mode="r", zarr_format=3)
        self.feat_array = zarr_store['merged_features']
        self.mapping_features = zarr_store['merged_features'].attrs["feature_columns"]

        self.ndvi = zarr_store['ndvi']
        self.ndsi = zarr_store['ndsi']
        self.dataset_len = self.ndvi.shape[0]
        self.timesteps = self.ndvi.shape[1]

        dates = zarr_store['dates'][:]
        self.dates = pd.to_datetime([d.decode('utf-8') for d in dates])
        dtindex = pd.DatetimeIndex(self.dates)
        self.doy = dtindex.dayofyear.to_numpy()
        is_leap = dtindex.is_leap_year.astype(int)
        self.t = (self.doy - 1) / (365 + is_leap)

        self.num_features = [
            f for f in self.features if f not in ['tree_species', 'habitat']
        ]
        self.nr_num_features = len(self.num_features)
        self.num_feature_indices = [i for f in self.num_features for i in self.mapping_features[f]]
        self.nr_tree_species = 17
        self.nr_habitats = 46

        self.missingness = zarr_store['missingness'][:]

    def __getitem__(self, idx):
        ndvi = torch.from_numpy(self.ndvi[idx])
        ndsi = torch.from_numpy(self.ndsi[idx])
        features = torch.from_numpy(self.feat_array[idx]).float()
        return ndvi, ndsi, features

    def __len__(self):
        return self.dataset_len

