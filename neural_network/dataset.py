import zarr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

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
        self.nr_tree_species = 18
        self.nr_habitats = 46

        self.missingness = zarr_store['missingness'][:]

    def __getitem__(self, idx):
        ndvi = torch.from_numpy(self.ndvi[idx])
        ndsi = torch.from_numpy(self.ndsi[idx])
        features = torch.from_numpy(self.feat_array[idx]).float()
        return ndvi, ndsi, features

    def __len__(self):
        return self.dataset_len

