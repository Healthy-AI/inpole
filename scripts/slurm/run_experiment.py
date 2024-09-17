import argparse

from amhelpers.config_parsing import load_config
from amhelpers.sweep import Sweep

try:
    from inpole import ESTIMATORS
    estimators_kwargs = {'choices': ESTIMATORS, 'default': ESTIMATORS}
except ModuleNotFoundError:
    estimators_kwargs = {}


TRAIN_SCRIPT_PATH = 'scripts/slurm/training_template'
PRE_SCRIPT_PATH = 'scripts/slurm/preprocessing_template'
POST_SCRIPT_PATH = 'scripts/slurm/postprocessing_template'

ACCOUNT = 'NAISS2023-22-686'

CONTAINER_PATH = '/mimer/NOBACKUP/groups/inpole/singularity/inpole_env.sif'


def custom_type(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--estimators', nargs='+', type=str, **estimators_kwargs)
    parser.add_argument('--train_script', type=str, default=TRAIN_SCRIPT_PATH)
    parser.add_argument('--pre_script', type=str, default=PRE_SCRIPT_PATH)
    parser.add_argument('--post_script', type=str, default=POST_SCRIPT_PATH)
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--n_hparams', type=int, default=10)
    parser.add_argument('--include_default_hparams', action='store_true')
    parser.add_argument('--account', type=str, default=ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['T4', 'V100', 'A40', 'A100'], default='A40')
    parser.add_argument('--container_path', type=str, default=CONTAINER_PATH)
    parser.add_argument('--sweep_param', type=str)
    parser.add_argument('--sweep_param_values', nargs='+', type=custom_type)
    parser.add_argument('--sweep_param_str', type=str)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config_path)

    sweep = Sweep(
        config,
        args.estimators,
        args.train_script,
        args.pre_script,
        args.post_script,
        args.sweep_param,
        args.sweep_param_values,
        args.sweep_param_str,
        args.n_trials,
        args.n_hparams,
        args.include_default_hparams,
        options={
            'account': args.account,
            'gpu': args.gpu,
            'container_path': args.container_path
        }
    )
    sweep.prepare()
    if not args.dry_run:
        sweep.launch()
