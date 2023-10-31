import argparse
import joblib
from os.path import join

from amhelpers.config_parsing import load_config
from amhelpers.amhelpers import save_yaml

from inpole import ESTIMATORS
from inpole.utils import create_results_dir_from_config
from inpole.train_predict import train, predict
from inpole.data import get_data_handler_from_config


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
    metrics = [
        ('accuracy', {}),
        ('auc', {'average': 'macro'}),
        ('auc', {'average': None}),
        ('ece', {}),
        ('sce', {})
    ]
    for subset in subsets:
        predict(config, pipeline, args.estimator, subset,
                metrics=metrics)
    
    if config['experiment'] == 'ra':
        for subset in subsets:
            predict(config, pipeline, args.estimator, subset,
                    metrics=metrics, switches_only=True)
        
    
    """if args.estimator == 'sdt' or args.estimator =='rdt':
        preprocessor, estimator = pipeline.named_steps.values()

        categories = preprocessor.get_feature_names_out()
        categories = [s.split('__')[1] for s in categories]

        estimator.align_axes()
        estimator.save_tree(categories, suffix='_before_pruning')

        data_handler = get_data_handler_from_config(config)
        X_valid, y_valid = data_handler.get_splits()[1]
        X_valid = preprocessor.transform(X_valid)
        y_valid = estimator.label_encoder_.transform(y_valid)
        dataset_valid = estimator.get_dataset(X_valid, y_valid)

        estimator.prune_tree(dataset_valid)
        estimator.save_tree(categories, suffix='_after_pruning')

        for subset in subsets:
            predict(config, pipeline, f'{args.estimator}_postprocessed',
                    subset, metrics=metrics)"""