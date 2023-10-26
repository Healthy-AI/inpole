import os
import argparse
import subprocess
from os.path import join

from inpole import ESTIMATORS
from inpole.sweep import create_jobscript_from_template

TRAIN_SCRIPT_PATH = 'scripts/slurm_templates/training'

ACCOUNT = 'NAISS2023-22-686'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--estimators', nargs='+', type=str, choices=ESTIMATORS, default=ESTIMATORS)
    parser.add_argument('--train_script', type=str, default=TRAIN_SCRIPT_PATH)
    parser.add_argument('--account', type=str, default=ACCOUNT)
    parser.add_argument('--gpu', type=str, choices=['T4', 'V100', 'A40', 'A100'], default='A40')
    args = parser.parse_args()

    jobdir = join(args.experiment_path, 'jobscripts')

    for estimator in args.estimators:
        command = ['sbatch']

        configs = os.listdir(join(args.experiment_path, 'configs', estimator))
        n_jobs = len(configs)
        if n_jobs > 1:
            command.append(f'--array=1-{n_jobs}')
        
        jobname = f'job_predict_{estimator}'
        jobscript_path = create_jobscript_from_template(
            template=args.train_script, experiment='',
            experiment_path=args.experiment_path, estimator=estimator,
            jobname=jobname, jobdir=jobdir,
            options={'account': args.account, 'gpu': args.gpu})
        
        with open(jobscript_path, 'r') as f:
            jobscript = f.read()
        
        jobscript = jobscript.strip() + ' --predict_only\n'
        
        with open(jobscript_path, 'w') as f:
            f.write(jobscript)
        
        command.append(jobscript_path)
        
        subprocess.run(command)
