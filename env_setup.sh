#!/bin/bash

# Activate the base environment.
conda activate

# Remove the environment if it already exists.
env_name=$(head -1 environment.yml | cut -d' ' -f2)
if conda info --envs | grep -q $env_name; then conda remove -y --name $env_name --all; fi

# Create the environment.
conda env create -f environment.yml
conda activate $env_name

# Install risk-slim and AIX360 in editable mode.
#
# Note: It is not possible to install these packages from the environment.yml
# file while controlling the installation path.
pip install -e git+https://github.com/antmats/risk-slim.git#egg=riskslim
pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[rbm]

# Install other packages using Poetry.
poetry install
