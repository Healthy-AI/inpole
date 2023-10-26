# Interpretable Policy Learning

This repository contains code to train and evaluate models for interpretably policy learning.

## Installation

```bash
$ git clone https://github.com/antmats/inpole.git
$ cd inpole
$ conda env create
$ conda activate inpole_env
$ poetry install
```

## Datasets

- Rheumatoid arthritis (RA)
- ...

## Train and evaluate a single model

Training and evaluating of a single model is handled via the script `scripts/train_predict.py`. Type `python scripts/train_predict.py -h` for details.

For example, to train a soft decicion tree (SDT) on the RA data, type:
```bash
$ python scripts/train_predict.py --config_path configs/ra.yaml --estimator sdt --new_out_dir
```

## Launching a sweep

A parameter sweep can be performed via the script `scripts/run_experiment.py`. Type `python scripts/run_experiment.py -h` for details.

For example, to sweep over soft decision trees (SDTs) on the RA data, type:
```bash
$ python scripts/run_experiment.py \
> --config_path configs/ra.yaml \
> --estimators SDT
> --account <ACCOUNT> \
> --gpu A40
```

## Alvis usage

Assuming the project is located in the folder `inpole`, a working Python environment utilizing pre-installed modules can be created in the following way:
```bash
$ cd inpole
$ module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 \
> scikit-learn/1.1.2-foss-2022a \
> IPython/8.5.0-GCCcore-11.3.0 \
> Seaborn/0.12.1-foss-2022a
$ virtualenv --system-site-packages inpole_env
$ source inpole_env/bin/activate
$ pip install --no-cache-dir --no-build-isolation \
> gitpython==3.1.32 skorch==0.13.0 amhelpers==0.4.3 \
> conda-lock==2.0.0 graphviz==0.20.1
$ pip install --no-cache-dir --no-build-isolation --no-deps -e .
```

### Using configs written for Alvis usage on your local computer

To automatically convert pathnames on Alvis to pathnames on your local computer, type the following commands which assume that you are located into the project folder and that data are located under `inpole/data` and results are saved under `inpole/results`:
```bash
$ export LOCAL_HOME_PATH=$HOME
$ export CLUSTER_PROJECT_PATH='/mimer/NOBACKUP/groups/inpole'
$ export LOCAL_PROJECT_PATH=$PWD
```

### Launch a Jupyter notebook

To launch a Jupyter notebook on Alvis, first crate a config file `inpole.sh` under `~/portal/jupyter/` and add the following lines:
```sh
cd $HOME/inpole

module purge

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-learn/1.1.2-foss-2022a
module load IPython/8.5.0-GCCcore-11.3.0
module load Seaborn/0.12.1-foss-2022a
module load JupyterLab/3.5.0-GCCcore-11.3.0

source inpole_env/bin/activate

python -m ipykernel install --user --name=inpole

jupyter lab --config="${CONFIG_FILE}"
```

Then, start a Jupyter notebook server on [Open OnDemand](https://portal.c3se.chalmers.se/public/root/) using the environment `~/portal/jupyter/inpole.sh`.

You can also find more information [here](https://www.c3se.chalmers.se/documentation/alvis-ondemand/#interactive-apps).
