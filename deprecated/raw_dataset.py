from typing import Union
import os
from pathlib import Path
import numpy as np
import xarray as xr


variables = {
    "s2_bands": ["s2_ndvi"],  # "s2_B02", "s2_B03", "s2_B04", "s2_B08"
}
FOREST_THRESH = 0.8


class RawDataset:
    def __init__(
        self, folder: Union[Path, str], variables=variables, fp16=False
    ):
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.filepaths = sorted(list(folder.glob("*_raw.nc")))
        self.type = np.float16 if fp16 else np.float32
        self.variables = variables

    def __getitem__(self, idx: int) -> dict:
        # Open the minicube
        filepath = self.filepaths[idx]
        minicube = xr.open_dataset(filepath, engine="h5netcdf")

        missing_dates = self.check_missing_timestamps(minicube)
        if missing_dates:
            # print(f"Inserting missing timestamps: {missing_dates}")
            minicube = minicube.reindex(
                time=np.sort(np.concatenate([minicube.time.values, missing_dates]))
            )

        # # Select the days with available data
        # indexes_avail = np.where(minicube.s2_avail.values == 1)[0]
        # # s2 is every 5 days
        # time = [minicube.time.values[i] for i in range(4, 450, 5)]
        # dates = [minicube.time.values[i] for i in (indexes_avail)]

        # print(indexes_avail)
        # print(time)
        # print(dates)

        # # Condition to check that the every 5 days inclus all the dates available (+ missing day)
        # if not set(dates) <= set(time):
        #     raise AssertionError(
        #         "ERROR: time indexes of the minicubes are not consistant ", filepath
        #     )
        
        # Create the minicube
        # Sentinel-2: s2 is 10 to 5 days, and already rescaled [0, 1]
        # s2_cube = (
        #     minicube[self.variables["s2_bands"]]
        #     .where((minicube.s2_mask == 0))  #  & minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7]))  # cloud cleaning
        #     .to_array()
        #     .values.transpose((1, 0, 2, 3))
        #     .astype(self.type)
        # )  # shape: (time, channels, h, w)

        # s2_scl = minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7]).values

        # s2_mask = (
        #     (minicube.FOREST_MASK.values > FOREST_THRESH)
        # )

        # timestamps = minicube.time.values

        # # Final minicube
        # data = {
        #     "eo": s2_cube,
        #     "forest_mask": s2_mask,
        #     "timestamps": timestamps,
        #     "cubename": os.path.basename(filepath),
        #     "scl": s2_scl,
        # }

        return minicube, os.path.basename(filepath)

    def __len__(self) -> int:
        return len(self.filepaths)
    
    def check_missing_timestamps(self, cube, max_conseq_dates=2):
        """Check for missing timestamps in cube.

        Args:
            cube (xr.Dataset): Cube to check for missing timestamps.
            max_conseq_dates (int): Maximum number of consecutive missing timestamps to allow.

        Returns:
            missing_dates (list): List of missing timestamps
        """
        timestamps = cube.time.values
        missing_dates = []
        current_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]
        nr_conseq_dates_max = 0
        while current_timestamp < last_timestamp:
            # Check for presence of next timestamp at 5 days interval
            expected_date = current_timestamp + np.timedelta64(5, "D")
            if expected_date not in timestamps:
                missing_dates.append(expected_date)
                # Record number of consecutive missing timestamps
                if len(missing_dates) > 1 and (
                    missing_dates[-1] - missing_dates[-2]
                ) == np.timedelta64(5, "D"):
                    nr_conseq_dates_max += 1
                else:
                    nr_conseq_dates_max = 1
            current_timestamp = expected_date

        if nr_conseq_dates_max > max_conseq_dates:
            print(f"Warning: Too many consecutive missing dates ({nr_conseq_dates_max})")

        return missing_dates
