import os
import argparse
import subprocess
from os.path import join

from amhelpers.sweep import create_jobscript_from_template

try:
    from inpole import ESTIMATORS
    estimators_kwargs = {'choices': ESTIMATORS, 'default': ESTIMATORS}
except ModuleNotFoundError:
    estimators_kwargs = {}

TRAIN_SCRIPT_PATH = 'scripts/slurm/training_template'

ACCOUNT = 'NAISS2023-22-686'

CONTAINER_PATH = '/mimer/NOBACKUP/groups/inpole/singularity/inpole_env.sif'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--estimators', nargs='+', type=str, **estimators_kwargs)
    parser.add_argument('--train_script', type=str, default=TRAIN_SCRIPT_PATH)
    parser.add_argument('--account', type=str, default=ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['T4', 'V100', 'A40', 'A100'], default='A40')
    parser.add_argument('--container_path', type=str, default=CONTAINER_PATH)
    args = parser.parse_args()

    jobdir = join(args.experiment_path, 'jobscripts')

    for estimator in args.estimators:
        command = ['sbatch']

        configs = os.listdir(join(args.experiment_path, 'configs', estimator))
        n_jobs = len(configs)
        if n_jobs > 1:
            command.append(f'--array=1-{n_jobs}')
        
        jobname = f'job_predict_{estimator}'
        options = {
            'account': args.account,
            'gpu': args.gpu,
            'container_path': args.container_path
        }
        jobscript_path = create_jobscript_from_template(
            template=args.train_script, experiment='',
            experiment_path=args.experiment_path, estimator=estimator,
            jobname=jobname, jobdir=jobdir,
            options=options
        )
        
        with open(jobscript_path, 'r') as f:
            jobscript = f.read()

        old = '--estimator $estimator'
        new = old + ' --predict_only'
        jobscript = jobscript.replace(old, new)
        
        with open(jobscript_path, 'w') as f:
            f.write(jobscript)
        
        command.append(jobscript_path)
        
        subprocess.run(command)
