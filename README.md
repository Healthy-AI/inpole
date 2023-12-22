# Interpretable Policy Learning

This repository contains code to train and evaluate models for interpretable policy learning.

## Installation

This package makes use of [risk-slim](https://github.com/ustunb/risk-slim), which in turn requires [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). Create an IBM account and install CPLEX by following [this link](https://www.ibm.com/account/reg/us-en/signup?formid=urx-20028).

Another dependency is an algorithm for learning falling rule lists (FRL) implemented in the [FRLOptimization](https://github.com/cfchen-duke/FRLOptimization) repository. The FRL algorithm requires FP-growth, a program that finds frequent item sets using the FP-growth algorithm and can be installed from [here](https://borgelt.net/fpgrowth.html).

To install `inpole`, type 
```bash
bash -l env_setup.sh
```
If the installation fails, make sure that [GCC](https://gcc.gnu.org/) is installed on your computer.

## Models

The following models are currently supported:
- soft decision tree (sdt)
- recurrent decision tree (rdt)
- feedforward prototype network (pronet)
- recurrent prototype network (prosenet)
- logistic regression (lr)
- hard decision tree (dt)
- [risk scores](https://github.com/antmats/risk-slim) (riskslim)
- [rule ensembles](https://github.com/christophM/rulefit/tree/master) (rulefit)
- [fast risk scores](https://github.com/jiachangliu/FasterRisk/tree/main) (fasterrisk)
- [falling rule lists](https://github.com/cfchen-duke/FRLOptimization/tree/master) (frl).

### Adding a new model

If you want to add a new model, please note the following:
- All models should be defined in [`inpole/models/models.py`](inpole/models/models.py). 
- Hyperparameters for all models should be specified in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py).
- All models should be given an alias (in brackets above) and listed in [`inpole/__init__.py`](inpole/__init__.py).
- All models should inherit from [`inpole.models.models.ClassifierMixin`](https://github.com/antmats/inpole/blob/main/inpole/models/models.py#L73) which enables evaluation with respect to several metrics (accuracy, AUC, ECE, SCE).
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
- Alzheimer's disease neuroimaging initiative (adni)
- treatment switching in rheumatoid arthritis (switch).

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
python scripts/train_predict.py \
    --config_path configs/example_config.yaml \
    --estimator sdt \
    --new_out_dir
```

## Alvis usage

On Alvis, the code can be run using a container. To create a container, copy the file [`container.def`](container.def) to a storage directory with plenty of space. Then, assuming the `inpole` repository is cloned to your home directory and you are located in the storage directory, type
```bash
apptainer build --bind $HOME:/mnt inpole_env.sif container.def
```
To run Python within the container, type
```bash
apptainer exec inpole_env.sif python --version
```

To access CPLEX from within the container, type
```bash
ml purge && ml CPLEX/22.1.1
export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
```

Example of how to train and evaluate a single model on Alvis:
```bash
container_path=/path/to/my/storage/directory/inpole_env.sif
account=my_project_name
cd $HOME/inpole
ml purge && ml CPLEX/22.1.1
export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
srun -A $account --gpus-per-node=T4:1 --pty bash
apptainer exec --nv $container_path python scripts/train_predict.py \
    --config_path configs/example_config.yaml \
    --estimator sdt \
    --new_out_dir
```
The flag `--nv` ensures that GPU resources can be accessed from within the container.

### Launching a sweep

A parameter sweep can be performed using the script [`scripts/run_experiment.py`](scripts/run_experiment.py). Type `apptainer exec $container_path python scripts/run_experiment.py -h` for details. This script uses the `sbatch` command, which unfortunately is not available from within the container, so a separate environment is needed to launch a sweep:
```bash
cd $HOME/inpole
ml purge && ml Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b PyYAML/6.0-GCCcore-12.2.0
virtualenv --system-site-packages sweep_env
source sweep_env/bin/activate
pip install --no-cache-dir --no-build-isolation gitpython~=3.1
pip install --no-cache-dir --no-build-isolation --no-deps amhelpers==0.5.4
pip install --no-cache-dir --no-build-isolation --no-deps -e .
```

By default, the script trains and evaluates 10 models with different hyperparameter choices for 5 different splits of the data. The number of hyperparameter choices and data splits can be controlled by the arguments `--n_hparams` and `--n_trials`, respectively. For each data split, the test performance (with respect to the metric(s) specified in the configuration file) for the model that performs best on the validation set is saved to a file `scores.csv` in the experiment folder. It is possible to repeat this process while changing an outer parameter using the arguments `--sweep_param` and `--sweep_param_values`. The value of `--sweep_param` should be a string that corresponds to a key in the configuration file; nested levels should be separated by "::". For example, to train a prototype model with a varying number of prototypes, say 5, 10 and 5, use `--sweep_param estimators::pronet::module__num_prototypes` and `sweep_param 5 10 15`.

Example of how to launch a sweep on Alvis:
```bash
cd $HOME/inpole
ml purge && ml Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b PyYAML/6.0-GCCcore-12.2.0
source sweep_env/bin/activate
python scripts/run_experiment.py \
    --config_path configs/example_config.yaml \
    --estimators sdt rdt
    --container_path $container_path
```

### Using configuration files written for Alvis usage on your local computer

To automatically convert pathnames on Alvis to pathnames on your local computer, type the following commands which assume that you are located in the `inpole` directory and that data are stored under `inpole/data` and results are saved under `inpole/results`:
```bash
export LOCAL_HOME_PATH=$HOME
export CLUSTER_PROJECT_PATH='/mimer/NOBACKUP/groups/inpole'
export LOCAL_PROJECT_PATH=$PWD
```

### Launch a Jupyter notebook

To launch a Jupyter notebook on Alvis, first create a configuration file `inpole.sh` in `~/portal/jupyter/` and add the following lines:
```sh
ml purge && ml CPLEX/22.1.1
export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

container_path=/path/to/my/storage/inpole_env.sif

# Use a bind mount to make code changes immediately available in the notebook.
apptainer exec --bind $HOME:/mnt --nv $container_path jupyter notebook --config="${CONFIG_FILE}"
```

Then, start a Jupyter notebook server on [Open OnDemand](https://portal.c3se.chalmers.se/public/root/) using the environment `~/portal/jupyter/inpole.sh`.

You can also find more information [here](https://www.c3se.chalmers.se/documentation/alvis-ondemand/#interactive-apps).
