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

## Models

The following models are currently supported:
- soft decision tree (sdt)
- recurrent decision tree (rdt)
- feedforward prototype network (pronet)
- recurrent prototype network (prosenet)
- logistic regression (lr)
- hard decision tree (dt).

All models are defined in [`inpole/models/models.py`](inpole/models/models.py) and their hyperparameters are specified in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py). Each model is also given an alias (in brackets above) and listed in [`inpole/__init__.py`](inpole/__init__.py).

Each model should inherit from `inpole.models.ClassifierMixin` to enable evalutation with respect to different metrics (accuracy, AUC, ECE, SCE). Furthermore, each model must implement the functions `fit`, `predict_proba` and `predict` as shown in the example below:

```python
from inpole.models.models import ClassifierMixin

class MyModel(ClassifierMixin):
    def fit(self, X, y, **fit_params):
        ...
    
    def predict_proba(self, X, y):
        ...
    
    def predict(self, X, y):
        ...
```

## Datasets

The following datasets are currently supported:
- Rheumatoid arthritis (ra).

All datasets are defined in [`inpole/data/data.py`](inpole/data/data.py). Data-dependent hyperparameters are specified in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py). Each dataset is also given an alias (in brackets above) and listed in [`inpole/__init__.py`](inpole/__init__.py).

## Train and evaluate a single model

Training and evaluating of a single model is handled via the script [`scripts/train_predict.py`](scripts/train_predict.py). Type `python scripts/train_predict.py -h` for details.

Example:
```bash
$ python scripts/train_predict.py --config_path configs/example_config.yaml --estimator sdt --new_out_dir
```

## Launching a sweep

A parameter sweep can be performed via the script [`scripts/run_experiment.py`](scripts/run_experiment.py). Type `python scripts/run_experiment.py -h` for details.

Example:
```bash
$ python scripts/run_experiment.py \
> --config_path configs/example_config.yaml \
> --estimators sdt \
> --account <ACCOUNT> \
> --gpu <GPU>
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
> conda-lock==2.0.0 graphviz==0.20.1 colorcet==3.0.1
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
