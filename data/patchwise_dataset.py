from typing import Callable, List, Union

import h5py
import torch
import torch.nn as nn


class PatchwiseDataset(torch.utils.data.Dataset):
    """DIMPEO dataset."""

    def __init__(
        self,
        path: str,
        transforms: List[Union[Callable, nn.Module]] = [],
        pixelwise: bool = False,
        cache: bool = False,
    ) -> None:
        self.file_path = path
        self.pixelwise = pixelwise
        self.cache = cache
        self.transforms = transforms

        # self.labels = []
        self.dataset = None

        # Initial check (or for caching)
        with h5py.File(self.file_path, "r") as file:
            if self.pixelwise:
                # TODO: get height, width from file
                self.height = 128
                self.width = 128
                self.dataset_len = len(file.get("time")) * self.height * self.width
            else:
                self.dataset_len = len(file.get("time"))
            if cache:
                self.cached_data = file.get("data")[:]
                # self.cached_labels = list(file["labels"])

    def __getitem__(self, index):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
        # Why fill dataset at getitem rather than init?
        # Creates dataset first time getitem is called

        if self.cache:
            data = self.cached_data[index]
            for t in self.transforms:
                data = t(data)
            return data

        # Each worker (which are forked after the init) need to have their own file handle
        if self.dataset is None:
            file = h5py.File(self.file_path, "r")
            self.dataset = file.get("data")
            # self.labels = data.get("labels")

        if self.pixelwise:
            # decompose into image index + height/width index
            img_index = index // (self.height * self.width)
            height_index, width_index = divmod(
                index % (self.height * self.width), self.width
            )
            data = self.dataset[img_index, height_index, width_index]
        else:
            data = self.dataset[index]

        # label = self.labels[index]
        for t in self.transforms:
            data = t(data)

        return data

    def __len__(self):
        return self.dataset_len
