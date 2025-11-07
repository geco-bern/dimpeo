"""
Transpose NDVI anomaly data from (N, T) to (T, N) using Dask for efficient processing.
"""
import dask.array as da
from dask.distributed import Client, LocalCluster

SOURCE_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_temporal.zarr"
TRANSPOSED_ZARR = "/data_2/scratch/sbiegel/processed/ndvi_dataset_spatial.zarr"
DASK_LOCAL_DIRECTORY = "/data_2/scratch/sbiegel/dask_worker_space"

def transpose_zarr(source_zarr, target_zarr, component="ndvi"):
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        processes=True,
        memory_limit="10GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    )
    client = Client(cluster)

    src = da.from_zarr(source_zarr, component=component)
    N, T = src.shape

    # transpose to (T, N)
    dst = src.T

    dst_rechunked = dst.rechunk(chunks=(1, N))

    dst_rechunked.to_zarr(
        target_zarr,
        component=component,
        overwrite=True,
        compute=True
    )

if __name__ == "__main__":
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="anomalies")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="anomaly_scores")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="params_2/params_lower")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="params_2/params_median")
    transpose_zarr(SOURCE_ZARR, TRANSPOSED_ZARR, component="params_2/params_upper")