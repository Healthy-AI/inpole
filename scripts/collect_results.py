import argparse
from os.path import join
from functools import partial 

from amhelpers.sweep import Postprocessing
from amhelpers.config_parsing import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    args = parser.parse_args()

    def score_sorter(df, metric):
        estimator = df.estimator_name[0]
        if estimator in ['sdt', 'rdt', 'truncated_rdt']:
            return df[
                (df.subset == 'valid') &
                (df.estimator_name == f'{estimator}_aligned')
            ][metric].item()
        return df[df.subset == 'valid'][metric].item()
    
    config_path = join(args.experiment_path, 'default_config.yaml')
    config = load_config(config_path)

    sort_by = config['results']['sort_by']
    sorter = partial(score_sorter, metric=sort_by)
    
    postprocessing = Postprocessing(args.experiment_path)
    postprocessing.collect_results(sorter)
    postprocessing.remove_files()
