# DIMPEO (Detecting Drought Impacts on Forests in Earth Observation Data)

This is the code for the DIMPEO project (SDSC collaboration) on forest anomaly detection from remote sensing data.

NOTE: This code assumes that the minicubes have been downloaded using the [GECO EarthNet Minicuber](https://github.com/geco-bern/earthnet-minicuber) and stored in `CUBE_DIR` (see setup section below). It can therefore be seen as the second stage in the multi-stage processing pipeline consisting of [GECO EarthNet Minicuber](https://github.com/geco-bern/earthnet-minicuber) --> this repo --> [DSL Vegetation Anomalies Mapper](https://github.com/dsl-unibe-ch/vegetation-anomalies).

## Setup

First create a conda environment and install the necessary packages:
```bash
conda create -n dimpeo
conda activate dimpeo
conda install -c conda-forge numpy pandas scipy matplotlib notebook xarray h5netcdf tensorboard pyproj h5py dask scikit-learn folium
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install unfoldNd
pip install -e .
```

Then, define the following environment variables:
```bash
conda env config vars set CUBE_DIR=/data_2/dimpeo/cubes SAVE_DIR=/data_2/scratch/$USER/dimpeo
conda activate dimpeo
mkdir -p $SAVE_DIR
```
`CUBE_DIR` contains the raw minicubes downloaded using the [GECO EarthNet Minicuber](https://github.com/geco-bern/earthnet-minicuber), `SAVE_DIR` points to the directory for saving outputs / checkpoints / etc.

## Deploy: get zarr store from minicubes

Currently, a trained neural network model is available for detecting anomalies in 2023 (the model is trained on years 2017-2022). To generate the anomaly zarr store from the minicubes using that model, use the following command:
```bash
python neural_network/inference.py --name <run_id_or_name> --encoder-path /data_2/scratch/dbrueggemann/output/encoder_nolon_era_500k.pt
```
`<run_id_or_name>` should be a unique id or name for that run. This process will take several hours. The anomaly zarr store will be saved under `$SAVE_DIR/anomalies_<run_id_or_name>.zarr`.

## Develop: train / test / develop the neural network

### Training 

First, the dataset has to be extracted by running
```bash
python neural_network/extract_dataset.py
```

Next, train the network with
```bash
python neural_network/train.py --name <run_id_or_name>
```
`<run_id_or_name>` should be a unique id or name for that training run.

### Inference

Generate the .zarr store by running inference
```bash
python neural_network/inference.py --name <run_id_or_name> --encoder-path <path/to/encoder.pt>
```
`<run_id_or_name>` should be a unique id or name for that run. `<path/to/encoder.pt>` has to be the path to the encoder checkpoint file generated during training.

To fine-tune the anomaly threshold (currently 1.5, a good default value in the literature, yields around 5% anomalies) it might make sense to skip the discretization step by setting `discretize=False` as the argument to `consolidate()` for the anomalies in `inference.py`. Afterwards, `discretize_anomalies()` can be run manually with different thresholds.

### Exploration / evaluation notebooks

Here is an overview of the notebooks:
- `notebooks/open_zarr.ipynb`: a demo on how to open and use the zarr file
- `notebooks/check_quantiles.ipynb`: qualitative evaluation of the neural network quantile regression
- `notebooks/folium_map.ipynb`: alternative way to create html maps with folium; need to create .npy files first by running `inference.py` with `save_npy=True`
- `notebooks/quantile_random_forest.ipynb`: quantile random forest approach
- `notebooks/statistical_baseline.ipynb`: showcases the statistical baseline for anomaly detection


## TODO

- [ ] Retrain the model with more cubes (there are more extracted cubes now in `/data_2/dimpeo/cubes/out_failed`)
- [ ] Separate the pixels into a training and test dataset (spatially), to evaluate the spatial generalizability of the method.
- [ ] Train models by holding out different years (not just 2023).
- [ ] Explore different anomaly smoothing / aggregation methods: e.g. Gaussian smoothing, monthly aggregation, ... (see `consolidate()` in `neural_network/helpers.py`)
- [ ] Find better way to define dates in `get_dates()` in `neural_network/helpers.py`
