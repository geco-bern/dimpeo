# DIMPEO Code

## Setup

```bash
conda create -n dimpeo
conda activate dimpeo
conda install -c conda-forge numpy pandas scipy matplotlib notebook xarray h5netcdf tensorboard pyproj h5py dask scikit-learn folium
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install unfoldNd
pip install -e .
```

## How to run

First, define the following environment variables:
```
export CUBE_DIR=/data_2/dimpeo/cubes
export PROC_DIR=/data_2/scratch/$USER/processed
export SAVE_DIR=/data_2/scratch/$USER/output
```

The code for the neural network approach is under `neural_network`. First, the dataset has to be extracted by running
```bash
python neural_network/extract_dataset.py
```

Next, train the network with
```bash
python neural_network/train.py --name train_exp --features "dem,fc,fh,slope,easting,northing,twi,rugg,curv,press_mean,press_std,temp_mean,temp_std,precip_mean,precip_std"
```

Finally, generate the .zarr file by running inference
```bash
python neural_network/inference.py --name inf_exp --encoder-name encoder_train_exp.pt --features "dem,fc,fh,slope,easting,northing,twi,rugg,curv,press_mean,press_std,temp_mean,temp_std,precip_mean,precip_std"
```
To fine-tune the anomaly threshold (currently 1.5, a good default value in the literature, yields around 5% anomalies) it might make sense to skip the discretization step by setting `discretize=False` as the argument to `consolidate()` for the anomalies in `inference.py`. Afterwards, `discretize_anomalies()` can be run manually with different thresholds.

Here is an overview of the notebooks:
- `notebooks/open_zarr.ipynb`: a demo on how to open and use the zarr file
- `notebooks/check_quantiles.ipynb`: qualitative evaluation of the neural network quantile regression
- `notebooks/folium_map.ipynb`: alternative way to create html maps with folium; need to create .npy files first by running `inference.py` with `save_npy=True`
- `notebooks/quantile_random_forest.ipynb`: quantile random forest approach
- `notebooks/statistical_baseline.ipynb`: showcases the statistical baseline for anomaly detection


## TODO

- Retrain the model with more cubes (there are more extracted cubes now in `/data_2/dimpeo/cubes/out_failed`)
- Separate the pixels into a training and test dataset (spatially), to evaluate the spatial generalizability of the method.
- Train models by holding out different years (not just 2023).
- Try using median (or other temporal aggregation) instead of mean to compute monthly values (`use_median=True` in `consolidate()`).