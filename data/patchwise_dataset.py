from typing import Callable, List, Union

import h5py
import numpy as np
import torch
import torch.nn as nn


class PatchwiseDataset(torch.utils.data.Dataset):
    """DIMPEO dataset."""

    def __init__(
        self,
        path: str,
        transforms: List[Union[Callable, nn.Module]] = [],
        spatiotemporal_features: List[str] = ["s2_ndvi"],
        spatial_features: List[str] = ["slope", "easting", "twi"],
        pixelwise: bool = False,
    ) -> None:
        self.file_path = path
        self.pixelwise = pixelwise
        self.transforms = transforms
        self.spatiotemporal_features = spatiotemporal_features
        self.spatial_features = spatial_features

        # self.labels = []
        self.spatiotemporal_dataset = None
        self.spatial_dataset = None

        # Initial check (or for caching)
        with h5py.File(self.file_path, "r") as file:
            if self.pixelwise:
                self.dataset_len = len(file.get("meta/valid_pixel_idx"))
            else:
                self.dataset_len = len(file.get("temporal/time"))

    def __getitem__(self, index):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
        # Why fill dataset at getitem rather than init?
        # Creates dataset first time getitem is called

        # Each worker (which are forked after the init) need to have their own file handle
        if self.spatiotemporal_dataset is None:
            file = h5py.File(self.file_path, "r")
            self.spatiotemporal_dataset = file.get("spatiotemporal")
            self.spatial_dataset = file.get("spatial")
            if self.pixelwise:
                self.valid_pixel_idx = file.get("meta/valid_pixel_idx")
            # self.labels = data.get("labels")

        if self.pixelwise:
            img_idx, height_idx, width_idx = self.valid_pixel_idx[index]
            st_data = np.stack(
                [
                    self.spatiotemporal_dataset[n][img_idx, height_idx, width_idx]
                    for n in self.spatiotemporal_features
                ],
                axis=-1,
            )
            s_data = np.stack(
                [
                    self.spatial_dataset[n][img_idx, height_idx, width_idx]
                    for n in self.spatial_features
                ],
                axis=-1,
            )
        else:
            st_data = np.stack(
                [
                    self.spatiotemporal_dataset[n][index]
                    for n in self.spatiotemporal_features
                ],
                axis=-1,
            )
            s_data = np.stack(
                [self.spatial_dataset[n][index] for n in self.spatial_features],
                axis=-1,
            )
        data = {"spatiotemporal": st_data, "spatial": s_data}

        # label = self.labels[index]
        for t in self.transforms:
            data = t(data)

        return data

    def __len__(self):
        return self.dataset_len
