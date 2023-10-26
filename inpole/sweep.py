import os
import glob
import copy
import warnings
from os.path import join, isfile
from subprocess import check_output
from pathlib import Path

import git
import pandas as pd
import numpy as np
from amhelpers.amhelpers import save_yaml

from inpole.utils import seed_hash
from inpole.utils import create_results_dir_from_config


def _extract_jobid(output):
    output = output.decode('utf-8')
    jobid = int(output.split('job')[1])
    return jobid


def create_jobscript_from_template(
    template, experiment, experiment_path,
    estimator, jobname, jobdir=None, options={}):
    
    with open(template, 'r') as f:
        text = f.read()
    
    text = text.replace('<EXPERIMENT>', experiment)
    text = text.replace('<EXPERIMENT_PATH>', experiment_path)
    text = text.replace('<ESTIMATOR>', estimator)
    
    for key, value in options.items():
        text = text.replace('<%s>' % key.upper(), value)
    
    if jobdir is None:
        jobdir = experiment_path
    
    jobscript_path = join(jobdir, jobname)
    with open(jobscript_path, 'w') as f:
        f.write(text)
    
    return jobscript_path


class Sweep:
    def __init__(
        self, config, estimators, job_template_training,
        job_template_preprocessing, job_template_postprocessing,
        n_trials=5, n_hparams=10, include_default_hparams=True,
        options={'gpu': 'A40'}):
        
        self.path, self.config = create_results_dir_from_config(
            config, suffix='sweep', update_config=True)
        
        self.estimators = estimators
        
        self.job_template_training = job_template_training
        self.job_template_preprocessing = job_template_preprocessing
        self.job_template_postprocessing = job_template_postprocessing
        
        self.n_trials = n_trials
        self.n_hparams = n_hparams
        self.include_default_hparams = include_default_hparams
        
        self.options = options

        try:
            repo = git.Repo(search_parent_directories=True)
            with open(join(self.path, 'repo.tar'), 'wb') as f:
                repo.archive(f)
            git_info = {
                'branch': repo.active_branch,
                'hash': repo.head.object.hexsha
            }
            with open(join(self.path, 'gitinfo.txt'), 'w') as f:
                print(git_info, file=f)
        except Exception as e:
            warnings.warn(f"Getting Git info failed: {e}.")

    def prepare(self):
        experiment = self.config['experiment']

        logs_dir = join(self.path, 'logs')
        Path(logs_dir).mkdir()

        # =====================================================================
        # Create a config file for each job.
        # =====================================================================
        
        save_yaml(self.config, self.path, 'default_config')

        trials_range = range(1, self.n_trials+1)
        hparams_range = range(self.n_hparams) if self.include_default_hparams \
            else range(1, self.n_hparams+1)

        for estimator in self.estimators:
            config_dir = join(self.path, 'configs', estimator)
            Path(config_dir).mkdir(parents=True)
            i_config = 1

            for trial_seed in trials_range:
                for hparams_seed in hparams_range:
                    config = copy.deepcopy(self.config)
                    
                    config['data']['seed'] = trial_seed
                    config['hparams']['seed'] = hparams_seed
                    config['estimators']['seed'] = seed_hash(
                        experiment, estimator, trial_seed, hparams_seed)
                    
                    results_path = join(
                        config['results']['path'], 'sweep',
                        f'trial_{trial_seed:02d}',
                        f'{estimator}_{hparams_seed:02d}')
                    config['results']['path'] = results_path
                    
                    Path(results_path).mkdir(parents=True)
                    save_yaml(config, results_path, 'config')

                    save_yaml(config, config_dir, f'config{i_config:03d}')
                    i_config += 1
        
        self._n_jobs = len(trials_range) * len(hparams_range)
    
        # =====================================================================
        # Create jobscripts.
        # =====================================================================

        jobscripts_dir = join(self.path, 'jobscripts')
        Path(jobscripts_dir).mkdir()

        self.jobsripcts = {'pre': None, 'main': {}, 'post': None}

        kwargs = {
            'experiment': experiment, 'experiment_path': self.path,
            'jobdir': jobscripts_dir, 'options': self.options
        }
        
        jobscript_path = create_jobscript_from_template(
            template=self.job_template_preprocessing,
            estimator='', jobname='job_pre', **kwargs)
        self.jobsripcts['pre'] = jobscript_path

        for estimator in self.estimators:
            jobname = f'job_{estimator}'
            jobscript_path = create_jobscript_from_template(
                template=self.job_template_training,
                estimator=estimator, jobname=jobname, **kwargs)
            self.jobsripcts['main'][estimator] = jobscript_path

        jobscript_path = create_jobscript_from_template(
            template=self.job_template_postprocessing,
            estimator='', jobname='job_post', **kwargs)
        self.jobsripcts['post'] = jobscript_path
    
    def launch(self):
        main_dependencies = [self._submit_job(self.jobsripcts['pre'])]
        
        post_dependencies = []
        for estimator in self.estimators:
            jobid = self._submit_job(
                self.jobsripcts['main'][estimator],
                main_dependencies, self._n_jobs)
            post_dependencies.append(jobid)
        
        self._submit_job(self.jobsripcts['post'], post_dependencies)

    def _submit_job(self, jobscript_path, dependencies=None, n_jobs=1):
        command = ['sbatch']
        if dependencies is not None:
            dependencies = [str(d) for d in dependencies]
            dependencies = ':'.join(dependencies)
            command.append(f'--dependency=afterok:{dependencies}')
        if n_jobs > 1:
            command.append(f'--array=1-{n_jobs}')
        command.append(jobscript_path)
        return _extract_jobid(check_output(command))


class Postprocessing:
    def __init__(self, exp_path):
        self.exp_path = exp_path
        config_dir = join(exp_path, 'configs')
        self.estimators = os.listdir(config_dir)
    
    def _get_trial_dirs(self):
        sweep_dir = join(self.exp_path, 'sweep')
        trial_dirs = [
            join(sweep_dir, x) for x in os.listdir(sweep_dir)
            if 'trial' in x
        ]
        return sorted(trial_dirs)

    def _sort_scores(self, path, sorter):
        all_sorted_scores = {}
        
        for estimator in self.estimators:
            experiments = [
                x for x in os.listdir(path)
                if x.startswith(estimator)
            ]

            csvs, scores = [], []
            
            for e in experiments:
                p = join(path, e, 'scores.csv')
                if not isfile(p):
                    print(f"File {p} not found.")
                    continue
                
                csv = pd.read_csv(p)
                if not 'subset' in csv.columns:
                    raise ValueError(
                        f"Column 'subset' not found in {p}.")
                csv.insert(0, 'exp', e)
                csvs.append(csv)

                score = sorter(csv)
                if not isinstance(score, float):
                    raise TypeError(
                        "The output of the score sorting function must "
                        f"be a float, but got type {type(score).__name__}.")
                scores.append(score)
            
            if len(csvs) > 0:
                scores = np.array(scores)
                if np.isnan(scores).any():
                    scores[np.isnan(scores)] = -np.inf
                csvs = [csvs[i] for i in np.argsort(scores)[::-1]]
                sorted_scores = pd.concat(csvs, ignore_index=True)
                all_sorted_scores[estimator] = sorted_scores
        
        return all_sorted_scores

    def _collect_best_scores(self, sorted_scores):
        best_scores_list = []

        for i_trial in range(len(sorted_scores)):
            for _estimator, scores in sorted_scores[i_trial].items():
                best_exp = scores.exp.iloc[0]
                best_score = scores.groupby('exp').get_group(best_exp)
                trial = pd.DataFrame({'trial': len(best_score) * [i_trial+1]})
                best_score = pd.concat((trial, best_score), axis=1)
                best_scores_list.append(best_score)
 
        return best_scores_list

    def _concat_scores(self, sorted_scores):
        best_scores = self._collect_best_scores(sorted_scores)
        scores = pd.concat(best_scores, ignore_index=True)
        return scores

    def collect_results(self, score_sorter=None):
        if score_sorter is None:
            score_sorter = lambda csv: csv[csv.subset=='valid']['auc'].item()

        sorted_scores = []
        for trial_dir in self._get_trial_dirs():
            sorted_scores += [self._sort_scores(trial_dir, score_sorter)]
        
        scores = self._concat_scores(sorted_scores)
        
        scores.to_csv(join(self.exp_path, 'scores.csv'), index=False)

    def remove_files(self):
        scores_path = join(self.exp_path, 'scores.csv')
        if not isfile(scores_path):
            raise FileNotFoundError(
                f"File {scores_path} is required but was not found.")
        scores = pd.read_csv(scores_path)

        assert 'trial' in scores.columns
        assert 'exp' in scores.columns
        dirs_to_keep = [
            join(self.exp_path, 'sweep', f'trial_{trial:02d}', exp)
            for trial, exp in zip(scores.trial, scores.exp)
        ]

        trial_dirs = self._get_trial_dirs()
        all_dirs = [join(d, x) for d in trial_dirs for x in sorted(os.listdir(d))]

        dirs_to_delete = set(all_dirs) - set(dirs_to_keep)

        for d in dirs_to_delete:
            for f in glob.glob(join(d, '*.pt')):
                os.remove(f)
            for f in glob.glob(join(d, '*.pkl')):
                os.remove(f)
