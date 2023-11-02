import argparse

from amhelpers.config_parsing import load_config

from inpole import ESTIMATORS
from inpole.sweep import Sweep


TRAIN_SCRIPT_PATH = 'scripts/slurm_templates/training'
PRE_SCRIPT_PATH = 'scripts/slurm_templates/preprocessing'
POST_SCRIPT_PATH = 'scripts/slurm_templates/postprocessing'

ACCOUNT = 'NAISS2023-22-686'

CONTAINER_PATH = '/mimer/NOBACKUP/groups/inpole/singularity/inpole_env.sif'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--estimators', nargs='+', type=str, choices=ESTIMATORS, default=ESTIMATORS)
    parser.add_argument('--train_script', type=str, default=TRAIN_SCRIPT_PATH)
    parser.add_argument('--pre_script', type=str, default=PRE_SCRIPT_PATH)
    parser.add_argument('--post_script', type=str, default=POST_SCRIPT_PATH)
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--n_hparams', type=int, default=10)
    parser.add_argument('--include_default_hparams', action='store_true')
    parser.add_argument('--account', type=str, default=ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['T4', 'V100', 'A40', 'A100'], default='A40')
    parser.add_argument('--container_path', type=str, default=CONTAINER_PATH)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config_path)

    sweep = Sweep(
        config,
        args.estimators,
        args.train_script,
        args.pre_script,
        args.post_script,
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
