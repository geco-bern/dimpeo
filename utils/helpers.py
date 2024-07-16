import numpy as np
import pandas as pd


def get_doy(dates):
    return np.array(
            [
                pd.to_datetime(d).to_pydatetime().timetuple().tm_yday
                for d in dates
            ],
            dtype=int,
    )


def check_missing_timestamps(cube, start_year, end_year, max_conseq_dates=2):
    """Check for missing timestamps in cube.

    Args:
        cube (xr.Dataset): Cube to check for missing timestamps.
        max_conseq_dates (int): Maximum number of consecutive missing timestamps to allow.

    Returns:
        missing_dates (list): List of missing timestamps
    """
    timestamps = cube.time.values
    missing_dates = []

    # beginning of 2017
    current_timestamp = timestamps[0]
    while (current_timestamp - np.timedelta64(5, "D")).astype("datetime64[Y]").astype(
        int
    ) + 1970 >= start_year:
        current_timestamp -= np.timedelta64(5, "D")
        missing_dates.append(current_timestamp)

    # end of 2023
    current_timestamp = timestamps[-1]
    while (current_timestamp + np.timedelta64(5, "D")).astype("datetime64[Y]").astype(
        int
    ) + 1970 <= end_year:
        current_timestamp += np.timedelta64(5, "D")
        missing_dates.append(current_timestamp)

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