#!/bin/bash

# Activate the base environment.
conda activate

# Remove the inpole environment if it already exists.
env_name=$(head -1 environment.yml | cut -d' ' -f2)
if conda info --envs | grep -q $env_name; then conda remove -y --name $env_name --all; fi

# Create and activate the inpole environment.
conda env create -f environment.yml
conda activate $env_name

# Install risk-slim and AIX360 in editable mode.
#
# These packages cannot be installed using Poetry. We could install them from
# the environment.yml file, but then we would first need to clone the GitHub
# repositories to a local directory. It is not possible to install these 
# packages from the environment.yml file while controlling the installation 
# path using the --src option.
pip install -e git+https://github.com/antmats/risk-slim.git#egg=riskslim
pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[rbm]

# Install the remaining packages using Poetry.
poetry install
