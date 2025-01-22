# DIMPEO Code

## Setup

```bash
conda create -n dimpeo
conda activate dimpeo
conda install -c conda-forge numpy pandas scipy matplotlib notebook xarray h5netcdf tensorboard pyproj h5py dask scikit-learn
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
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