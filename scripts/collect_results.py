import argparse
from functools import partial 

from inpole.sweep import Postprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    args = parser.parse_args()

    def score_orter(df, metric):
        return df[df.subset == 'valid'][metric].item()

    sorter = partial(score_orter, metric='auc')
    
    postprocessing = Postprocessing(args.experiment_path)
    postprocessing.collect_results(sorter)
