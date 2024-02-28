# DIMPEO Code

## Setup

```bash
conda create -n dimpeo
conda activate dimpeo
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install torchgeo
conda install xarray
conda install h5netcdf
# pip install "jsonargparse[signatures]"
# conda install scikit-learn
# conda install tensorboard
pip install -e .
```