from datetime import datetime
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
        annual: bool = False,
        loc: bool = True,
    ) -> None:
        self.file_path = path
        self.pixelwise = pixelwise
        self.annual = annual
        self.transforms = transforms
        self.spatiotemporal_features = spatiotemporal_features
        self.spatial_features = spatial_features

        self.spatiotemporal_dataset = None
        self.spatial_dataset = None
        self.temporal_dataset = None
        self.loc = loc

        with h5py.File(self.file_path, "r") as file:
            if self.pixelwise:
                if self.annual:
                    self.dataset_len = len(file.get("meta/annual_pixel_idx"))
                else:
                    self.dataset_len = len(file.get("meta/pixel_idx"))
            else:
                if self.annual:
                    self.dataset_len = len(file.get("meta/annual_idx"))
                else:
                    self.dataset_len = len(file.get("temporal/time"))

        self.original_indices = np.arange(self.dataset_len)
        self.subset_indices = self.original_indices

    def __getitem__(self, index):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
        # Why fill dataset at getitem rather than init?
        # Creates dataset first time getitem is called

        # Each worker (which are forked after the init) need to have their own file handle
        if self.spatiotemporal_dataset is None:
            self.file = h5py.File(self.file_path, "r+")
            self.spatiotemporal_dataset = self.file.get("spatiotemporal")
            self.spatial_dataset = self.file.get("spatial")
            self.temporal_dataset = self.file.get("temporal")
            if self.pixelwise:
                if self.annual:
                    self.annual_pixel_idx = self.file.get("meta/annual_pixel_idx")
                else:
                    self.pixel_idx = self.file.get("meta/pixel_idx")
                self.drought_mask = self.file.get("spatiotemporal/drought_mask")
            else:
                if self.annual:
                    self.annual_idx = self.file.get("meta/annual_idx")
            if self.loc:
                self.lon = self.file.get("meta/longitude")
                self.lat = self.file.get("meta/latitude")
            time = self.file.get("temporal/time")
            self.time = np.array(time, dtype="datetime64")

        sel_index = self.original_indices[self.subset_indices[index]]

        if self.pixelwise:
            # will return samples in the following format:
            # spatiotemporal: B x T x C
            # spatial: B x C
            if self.annual:
                img_idx, height_idx, width_idx, start_t_idx, end_t_idx = (
                    self.annual_pixel_idx[sel_index]
                )
                st_data = np.stack(
                    [
                        self.spatiotemporal_dataset[n][
                            img_idx, start_t_idx : end_t_idx + 1, height_idx, width_idx
                        ]
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

                img_idx, height_idx, width_idx = self.pixel_idx[sel_index]
                st_data = np.stack(
                    [
                        self.spatiotemporal_dataset[n][
                            img_idx, :, height_idx, width_idx
                        ]
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
            if self.loc:
                lon = self.lon[img_idx, width_idx]
                lat = self.lat[img_idx, height_idx]
        else:
            # will return samples in the following format:
            # spatiotemporal: B x T x H x W x C
            # spatial: B x H x W x C
            if self.annual:
                img_idx, start_t_idx, end_t_idx = self.annual_idx[sel_index]
                st_data = np.stack(
                    [
                        self.spatiotemporal_dataset[n][
                            img_idx, start_t_idx : end_t_idx + 1
                        ]
                        for n in self.spatiotemporal_features
                    ],
                    axis=-1,
                )
                s_data = np.stack(
                    [self.spatial_dataset[n][img_idx] for n in self.spatial_features],
                    axis=-1,
                )
            else:
                img_idx = sel_index
                st_data = np.stack(
                    [
                        self.spatiotemporal_dataset[n][img_idx]
                        for n in self.spatiotemporal_features
                    ],
                    axis=-1,
                )
                s_data = np.stack(
                    [self.spatial_dataset[n][img_idx] for n in self.spatial_features],
                    axis=-1,
                )
            if self.loc:
                lon = self.lon[img_idx, :]
                lat = self.lat[img_idx, :]

        dgs = self.convert_date_to_dgs(self.temporal_dataset["time"][img_idx])

        data = {"spatiotemporal": st_data, "spatial": s_data, "dgs": dgs}
        if self.loc:
            data["lon"] = lon
            data["lat"] = lat

        for t in self.transforms:
            data = t(data)

        return data

    def __len__(self):
        return self.dataset_len

    def inspect_file(self):
        def scan_node(g, tabs=0):
            print(" " * tabs, g.name)
            for v in g.values():
                if isinstance(v, h5py.Dataset):
                    print(" " * tabs + " " * 2 + " -", v.name)
                elif isinstance(v, h5py.Group):
                    scan_node(v, tabs=tabs + 2)

        with h5py.File(self.file_path, "r") as f:
            scan_node(f)

    @staticmethod
    def convert_date_to_dgs(dates):
        return np.array(
            [
                datetime.strptime(d[:10], "%Y-%m-%d").timetuple().tm_yday
                for d in dates.astype("U29")
            ],
            dtype=int,
        )

    def update_indices(self, new_indices):
        self.subset_indices = new_indices
        self.dataset_len = len(new_indices)
