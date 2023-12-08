import argparse
import joblib
from os.path import join

from amhelpers.config_parsing import load_config
from amhelpers.amhelpers import save_yaml
from amhelpers.amhelpers import create_results_dir_from_config

from inpole import ESTIMATORS
from inpole.train_predict import train, predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--estimator', type=str, choices=ESTIMATORS, required=True)
    parser.add_argument('--new_out_dir', action='store_true')
    parser.add_argument('--predict_only', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.new_out_dir:
        if args.predict_only:
            raise ValueError(
                "The arguments '--new_out_dir' and "
                "'--predict_only' cannot be used together.")
        results_path, config = create_results_dir_from_config(
            config, args.estimator, update_config=True)
        save_yaml(config, results_path, 'config')

    # Train or load model.
    f_pipeline = join(config['results']['path'], 'pipeline.pkl')
    if args.predict_only:
        pipeline = joblib.load(f_pipeline)
    else:
        pipeline = train(config, args.estimator)
        joblib.dump(pipeline, f_pipeline)
    
    # Evaluate model.
    subsets = ['valid', 'test']
    metrics = config['metrics']
    for subset in subsets:
        predict(config, pipeline, args.estimator, subset,
                metrics=metrics)
    
    if config['experiment'] == 'ra':
        for subset in subsets:
            predict(config, pipeline, args.estimator, subset,
                    metrics=metrics, switches_only=True)
