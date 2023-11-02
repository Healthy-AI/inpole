# Interpretable Policy Learning

This repository contains code to train and evaluate models for interpretably policy learning.

## Installation

This package makes use of [risk-slim](https://github.com/ustunb/risk-slim), which in turn requires [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). Install CPLEX by following the instructions [here](https://github.com/ustunb/risk-slim/blob/master/docs/cplex_instructions.md) (you can ignore step 4). Then, clone the [risk-slim](https://github.com/ustunb/risk-slim) repository to your local computer:
```bash
$ git clone https://github.com/ustunb/risk-slim.git
```

Now, install `inpole` by typing the following commands:
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
- hard decision tree (dt)
- [risk scores](https://github.com/ustunb/risk-slim/tree/master) (riskslim)
- [rule ensembles](https://github.com/christophM/rulefit/tree/master) (rulefit)
- [fast risk scores](https://github.com/jiachangliu/FasterRisk/tree/main) (fasterrisk).

### Adding a new model

If you want to add a new model, please note the following:
- All models should be defined in [`inpole/models/models.py`](inpole/models/models.py). 
- Hyperparameters for all models should be specified in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py).
- All models should be given an alias (in brackets above) and listed in [`inpole/__init__.py`](inpole/__init__.py).
- All models should inherit from [`inpole.models.models.ClassifierMixin`](https://github.com/antmats/inpole/blob/main/inpole/models/models.py#L73) which enables evalutation with respect to several metrics (accuracy, AUC, ECE, SCE).
- All models should implement the functions `fit`, `predict_proba` and `predict` as shown in the example below.

```python
import numpy as np

class MyModel(ClassifierMixin):
    def fit(self, X, y, **fit_params):
        # Infer classes from `y`.
        self.classes_ = np.unique(y)
        ...
    
    def predict_proba(self, X, y):
        ...
    
    def predict(self, X, y):
        ...
```

## Datasets

The following datasets are currently supported:
- rheumatoid arthritis (ra)
- alzheimer's disease neuroimaging initiative (adni).

### Adding a new dataset

If you want to add a new dataset, please note the following:
- All datasets should be defined in [`inpole/data/data.py`](inpole/data/data.py).
- Data-dependent hyperparameters should be specified in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py).
- All datasets should be given an alias (in brackets above) and listed in [`inpole/__init__.py`](inpole/__init__.py).

## Configuration files

Each dataset corresponds to an experiment with the same name as the dataset's alias (e.g., "ra"). Experiment-specific settings such as paths to data and results, evaluation metrics and seeds should be specified in a YAML file. See [`configs/example_config.yaml`](configs/example_config.yaml) for an example.

## Train and evaluate a single model

Training and evaluation of a single model is performed using the script [`scripts/train_predict.py`](scripts/train_predict.py). Type `python scripts/train_predict.py -h` for details.

Example:
```bash
$ python scripts/train_predict.py \
> --config_path configs/example_config.yaml \
> --estimator sdt \
> --new_out_dir
```

## Alvis usage

On Alvis, the code can be run using a container. To create a container, copy the file [`inpole_env.def`](`inpole_env.def) to a storage location with plenty of space. Then, assuming the `inpole` repository is cloned to your home directory and you are located in the storage directory, type
```bash
$ apptainer build --bind $HOME:/mnt inpole_env.sif inpole_env.def
```
To run Python within the container, type
```
$ apptainer exec inpole_env.sif python --version
```

Example of how to train and evaluate a single model on Alvis:
```bash
$ container_path=/path/to/my/storage/inpole_env.sif
$ account=my_project_name
$ cd $HOME/inpole
$ srun -A $account --gpus-per-node=T4:1 \
> apptainer exec --nv $container_path python scripts/train_predict.py \
> --config_path configs/example_config.yaml \
> --estimator sdt \
> --new_out_dir
```
The flag `--nv` ensures that GPU resources can be accessed from within the container.

### Launching a sweep

A parameter sweep can be performed using the script [`scripts/run_experiment.py`](scripts/run_experiment.py). Type `apptainer exec $container_path python scripts/run_experiment.py -h` for details. This script uses the `sbatch` command, which unfortunatelly is not available from within the container, so a separate environment is needed to launch the sweep:
```bash
cd $HOME/inpole
module purge && module load Python/3.10.8-GCCcore-12.2.0
virtualenv --system-site-packages sweep_env
source sweep_env/bin/activate
pip install --no-cache-dir --no-build-isolation amhelpers==0.5.1
pip install --no-cache-dir --no-build-isolation --no-deps -e .
```

Example of how to launch a sweep on Alvis:
```bash
cd $HOME/inpole
module purge && module load Python/3.10.8-GCCcore-12.2.0
source sweep_env/bin/activate
$ python scripts/run_experiment.py \
> --config_path configs/example_config.yaml \
> --estimators sdt rdt
> --container_path $container_path
```

### Using configuration files written for Alvis usage on your local computer

To automatically convert pathnames on Alvis to pathnames on your local computer, type the following commands which assume that you are located in the `inpole` directory and that data are stored under `inpole/data` and results are saved under `inpole/results`:
```bash
$ export LOCAL_HOME_PATH=$HOME
$ export CLUSTER_PROJECT_PATH='/mimer/NOBACKUP/groups/inpole'
$ export LOCAL_PROJECT_PATH=$PWD
```

### Launch a Jupyter notebook

To launch a Jupyter notebook on Alvis, first crate a configuration file `inpole.sh` in `~/portal/jupyter/` and add the following lines:
```sh
module purge

container_path=/path/to/my/storage/inpole_env.sif

apptainer exec --nv $container_path jupyter notebook --config="${CONFIG_FILE}"
```

Then, start a Jupyter notebook server on [Open OnDemand](https://portal.c3se.chalmers.se/public/root/) using the environment `~/portal/jupyter/inpole.sh`.

You can also find more information [here](https://www.c3se.chalmers.se/documentation/alvis-ondemand/#interactive-apps).
