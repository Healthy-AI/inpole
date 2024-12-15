# Representing history for interpretable modeling of clinical policies

Modeling policies for sequential clinical decision-making based on observational data is useful for describing treatment practices, standardizing frequent patterns in treatment, and evaluating alternative policies. For each task, it is essential that the policy model is interpretable. Learning accurate models requires effectively capturing a patient’s state, either through sequence representation learning or carefully crafted summaries of their medical history. While recent work has favored the former, it remains a question as to how histories should best be represented for interpretable policy modeling.

This repository contains the code used for the experiments in [our paper](https://arxiv.org/abs/2412.07895), where we systematically compare various approaches to summarizing patient history for interpretable clinical policy modeling across four sequential decision-making tasks: Alzheimer’s disease (ADNI), rheumatoid arthritis (RA), sepsis, and chronic obstructive pulmonary disease (COPD).

## Installation

This code makes use of [risk-slim](https://github.com/ustunb/risk-slim), which in turn requires [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). Create an IBM account and install CPLEX by following [this link](https://www.ibm.com/account/reg/us-en/signup?formid=urx-20028).

Another dependency is an algorithm for learning falling rule lists (FRL) implemented in the [FRLOptimization](https://github.com/cfchen-duke/FRLOptimization) repository. The FRL algorithm requires FP-growth, a program that finds frequent item sets using the FP-growth algorithm and can be installed from [here](https://borgelt.net/fpgrowth.html).

To set up a working environment, run the following command:
```bash
bash -l env_setup.sh
```
If the installation fails, make sure that [GCC](https://gcc.gnu.org/) is installed on your computer.

## Configuration files

For each experiment (ADNI, RA, Sepsis, and COPD), a corresponding configuration file is provided in [`configs`](configs). These configuration files specify details such as the path to the dataset, the evaluation metrics to be used, and the directory where the results will be saved. Additionally, the configuration file defines how the state $S_t$, i.e., the model input, is constructed. Each state corresponds to a certain configuration of the config file, see the table below.

| State $S_t$              | `include_context_variables`   | `include_previous_treatment`    | `aggregate_history`  | `reduction`                   | `add_current_context` | `shift_periods` |
|--------------------------|-------------------------------|--------------------------------|-----------------------|-------------------------------|-----------------------|-----------------|
| $X_t$                    | `True`                        | `False`                        | `False`               | `'null'`                      | `False`               | 0               |
| $A_{t-1}$                | `False`                       | `True`                         | `False`               | `'null'`                      | `False`               | 0               |
| $H_{(t-0):t}$            | `True`                        | `True`                         | `False`               | `'null'`                      | `False`               | 0               |
| $\bar{H}_t$              | `True`                        | `True`                         | `True`                | `'sum'`,`'max'`, or `'mean'`  | `False`               | 0               |
| $H_{(t-0):t}, \bar{H}_t$ | `True`                        | `True`                         | `True`                | `'sum'`,`'max'`, or `'mean'`  | `True`                | 0               |
| $H_{(t-1):t}, \bar{H}_t$ | `True`                        | `True`                         | `True`                | `'sum'`,`'max'`, or `'mean'`  | `True`                | 1               |
| $H_{(t-2):t}, \bar{H}_t$ | `True`                        | `True`                         | `True`                | `'sum'`,`'max'`, or `'mean'`  | `True`                | 2               |
| $H_t$                    | `True`                        | `True`                         | `False`               | `'null'`                      | `False`               | 0               |

## Data

To prepare the datasets for the experiments, begin by collecting the input data as outlined in the instructions below. Once the input data is ready, execute the script [`scripts/make_datasets.py`](scripts/make_datasets.py) to generate the datasets required for the experiments. Be sure to update the data paths in the configuration files to reflect the correct locations. A brief description of each dataset is provided in the table below.

|                                   | ADNI              | RA                | Sepsis            | COPD              |
|-----------------------------------|-------------------|-------------------|-------------------|-------------------|
| **Patients, n**                   | 1,605             | 4,391             | 20,932            | 7,977             |
| **Age in years, median (IQR)**    | 73.9 (69.3, 78.8) | 58.0 (49.0, 66.0) | 66.1 (53.7, 77.9) | 67.0 (56.0, 77.0) |
| **Female, n (%)**                 | 715 (44.5)        | 3,355 (76.5)      | 9,250 (44.2)      | 3,472 (43.5)      |
| **Patient observations $X_t$, n** | 6                 | 33                | 18                | 37                |
| **Actions $A_t$, n**              | 2                 | 8                 | 25                | 25                |
| **Stages $T$, median (IQR)**      | 3.0 (3.0, 3.0)    | 5.0 (3.0, 8.0)    | 13.0 (10.0, 17.0) | 18.0 (18.0, 18.0) |

### ADNI

To use ADNI, you must first apply for access [here](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData). After gaining access, follow these steps:
1. Log in to the [Image and Data Archive](https://ida.loni.usc.edu/login.jsp).
2. Under "Select Study", choose "ADNI". Then choose "Download > Study Data" and search for "ADNIMERGE".
3. Download the file "ADNIMERGE - Key ADNI tables merged into one table - Packages for R [ADNI1,GO,2]".
4. Install the ADNIMERGE package for R by following the instructions [here](https://adni.bitbucket.io/index.html).
5. Load the data and save it to a CSV file by running the R script below:

```R
library(ADNIMERGE)
data <- adnimerge
write.csv(data, file='/path/to/my/adni/data.csv', row.names=FALSE)
```

### RA

The RA data were available from [CorEvitas, LLC](https://www.corevitas.com/registry/rheumatoid-arthritis/) through a commercial subscription agreement and are not publicly available.

### Sepsis

The Sepsis data were preprocessed as decribed in [Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5). To obtain the preprocessed dataset, follow the instructions provided [here](https://github.com/antmats/case_based_ope?tab=readme-ov-file#collect-sepsis-dataset).

### COPD

To be updated.

## Models

In our experiments, we consider three types of models for hand-crafted states, i.e., all states except $S_t=H_t$:
- decision tree (dt)  
- logistic regression (lr)
- multilayer perceptron (mlp).

For ADNI, we also include [risk scores](https://github.com/antmats/risk-slim) (riskslim), i.e., scoring systems that enable probabilistic predictions.

For the state $S_t=H_t$, we include three different models:
- recurrent decision tree (rdt)
- prototype sequence network (prosenet)
- recurrent neural network (rnn).

Hyperparameters for each model are defined in [`inpole/models/hparam_registry.py`](inpole/models/hparam_registry.py).

## Model training and evaluation

To train and evaluate a single model on a specific dataset, use the script [`scripts/train_predict.py`](scripts/train_predict.py). Type `python scripts/train_predict.py -h` for details.

**Example.** Fit a decision tree to the RA data:
```bash
python scripts/train_predict.py \
    --config_path configs/ra.yaml \
    --estimator dt \
    --new_out_dir
```

## Run experiments on a cluster

In the paper, we train multiple models for each state and dataset. Each such experiment can be run using SLURM with the script [`scripts/slurm/run_experiment.py`](scripts/slurm/run_experiment.py). Type `python scripts/slurm/run_experiment.py -h` for details. 

The script creates a new experiment folder and initiates several tasks: a preprocessing job (currently unused), a set of main jobs, and a postprocessing job. In the main jobs, candidate models are trained and evaluated using randomly sampled hyperparameters. The postprocessing job then identifies the best models, and the results are compiled into a single file, `scores.csv`, which is saved in the experiment folder. There is a SLURM template script for each of these steps located in [`scripts/slurm`](scripts/slurm). You may need to modify these scripts based on your cluster's configuration. By default, the script trains and evaluates 10 candidate models for 5 different data splits. The number of hyperparameter configurations and data splits can be adjusted using the arguments `--n_hparams` and `--n_trials`, respectively.

You can repeat this process while varying an outer parameter using the`--sweep_param` and `--sweep_param_values` arguments. The value for `--sweep_param` should be a string corresponding to a key in the config file, with nested levels separated by ::. For example, to train a prototype sequence network with a varying number of prototypes (e.g., 5, 10, and 15), use `--sweep_param estimators::prosenet::module__num_prototypes` and `sweep_param 5 10 15`.

**Example.** Fit a logistic regression, a decision tree, and a multilayer perceptron to the RA data:
```bash
python scripts/run_experiment.py \
    --config_path configs/ra.yaml \
    --estimators lr dt mlp \
    --n_trials 5 \
    --n_hparams 5
```

## Alvis usage

On the [Alvis cluster](https://www.c3se.chalmers.se/about/Alvis/), you can run the code using a container. To create a container, copy the file [`container.def`](container.def) to a storage directory with ample space. Then, assuming the `inpole` repository is cloned to your home directory and you are located in the storage directory, run the following command:
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

**Example.** Fit a multilayer perceptron to the RA data on Alvis:
```bash
container_path=/path/to/my/storage/directory/inpole_env.sif
account=my_account
cd $HOME/inpole
ml purge && ml CPLEX/22.1.1
export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
srun -A $account --gpus-per-node=T4:1 --pty bash
apptainer exec --nv $container_path python scripts/train_predict.py \
    --config_path configs/example_config.yaml \
    --estimator mlp \
    --new_out_dir
```
The flag `--nv` ensures that GPU resources can be accessed from within the container.

### Run all experiments on the Alvis cluster

To reproduce all experiments, use the bash script [`scripts/slurm/run_experiments.sh`](scripts/slurm/run_experiments.sh). This script takes as input a config file and a text file specifiying all the state and model combinations to be considered. The file [`settings.csv`](settings.csv) contains the states and models used for for RA, Sepsis, and COPD. For ADNI, risk scores (riskslim) were also used for all states except $S_t=H_t$.

**Example.** Run all RA experiments on Alvis:
```bash
./scripts/slurm/run_experiments.sh settings.csv configs/ra.yaml
```

The script calls the Python script [`scripts/slurm/run_experiment.py`](scripts/slurm/run_experiment.py), which in turn uses the `sbatch` command to submit jobs to Alvis. Since `sbatch` is not available from within the container, a separate environment is needed. To create this environment, run the following commands:
```bash
cd $HOME/inpole
ml purge && ml Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b PyYAML/6.0-GCCcore-12.2.0
virtualenv --system-site-packages sweep_env
source sweep_env/bin/activate
pip install --no-cache-dir --no-build-isolation gitpython~=3.1
pip install --no-cache-dir --no-build-isolation --no-deps amhelpers==0.5.5
pip install --no-cache-dir --no-build-isolation --no-deps -e .
```

**Example.** Fit a logistic regression, a decision tree, and a multilayer perceptron to the RA data on Alvis:
```bash
cd $HOME/inpole
ml purge && ml Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b PyYAML/6.0-GCCcore-12.2.0
source sweep_env/bin/activate
python scripts/run_experiment.py \
    --config_path configs/example_config.yaml \
    --estimators lr dt mlp \
    --account $account \
    --container_path $container_path
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

### Automatic conversion of configuration files

To automatically convert pathnames from Alvis to your local computer, use the following commands, assuming you are in the inpole directory, with data stored under `inpole/data` and results saved under `inpole/results`:
```bash
export LOCAL_HOME_PATH=$HOME
export CLUSTER_PROJECT_PATH='/mimer/NOBACKUP/groups/inpole'
export LOCAL_PROJECT_PATH=$PWD
```

## Citation

To be updated.

## Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

The computations in this work were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973.
