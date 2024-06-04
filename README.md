# DIMPEO Code

## Setup

```bash
conda create -n dimpeo
conda activate dimpeo
conda install -c conda-forge lightning
conda install -c conda-forge matplotlib
conda install -c conda-forge xarray
conda install -c conda-forge h5netcdf
conda install -c conda-forge statsmodels
conda install -c conda-forge scikit-learn
# to use R (installing r-ks did not work on my Mac)
conda install -c conda-forge r-base
R
    > install.packages("ks", dependencies=TRUE)
    > q()
#
conda install -c conda-forge rpy2
# pip install "jsonargparse[signatures]"
# conda install tensorboard
pip install -e .
```