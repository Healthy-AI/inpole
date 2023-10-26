"""
Data handler for each experiment.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
#from scipy.stats import rankdata

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    #FunctionTransformer,
    #StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    LabelEncoder
)
from sklearn.compose import (
    #make_column_transformer,
    make_column_selector,
    ColumnTransformer
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer

from . import hparam_registry


__all__ = [
    'get_data_handler_class',
    'get_data_handler_from_config',
    'RAData'
]


def get_data_handler_class(name):
    if name not in globals():
        raise NotImplementedError(f"Data not found: {name}.")
    return globals()[name]


def get_data_handler_from_config(config):
    data_handler_name = config['experiment'].upper() + 'Data'
    return get_data_handler_class(data_handler_name)(**config['data'])


def _split_grouped_data(X, y, groups, valid_size, test_size, seed=None):
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    y = LabelEncoder().fit_transform(y)

    Xg = pd.concat([X, groups], axis=1)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    ii_train, ii_test = next(gss.split(X, y, groups))
    
    Xg_train, y_train = Xg.iloc[ii_train], y[ii_train]
    Xg_test, y_test = Xg.iloc[ii_test], y[ii_test]
    groups_train = groups.iloc[ii_train]

    if valid_size > 0:
        train_size = 1 - test_size
        _valid_size = valid_size / train_size
        
        gss = GroupShuffleSplit(n_splits=1, test_size=_valid_size, random_state=seed)
        ii_train, ii_valid = next(gss.split(Xg_train, y_train, groups_train))
        
        Xg_valid, y_valid = Xg_train.iloc[ii_valid], y_train[ii_valid]
        Xg_train, y_train = Xg_train.iloc[ii_train], y_train[ii_train]
    else:
        Xg_valid, y_valid = None, None

    data_train = (Xg_train, y_train)
    data_valid = (Xg_valid, y_valid)
    data_test = (Xg_test, y_test)

    return data_train, data_valid, data_test


class Data(ABC):
    def __init__(self, path, valid_size, test_size, seed, sample_size=None):
        self.path = path
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed
        self.sample_size = sample_size
    
    @property
    def alias(self):
        return type(self).__name__.replace('Data', '').lower()

    @property
    @abstractmethod
    def TREATMENT(self):
        pass

    @property
    @abstractmethod
    def GROUP(self):
        pass
    
    def _get_preprocessor_params(self, hparams_seed):
        if hparams_seed == 0:
            hparams = hparam_registry.default_hparams(self.alias)
        else:
            hparams = hparam_registry.random_hparams(self.alias, hparams_seed)
        return hparams
    
    @abstractmethod
    def get_preprocessing_steps(self):
        pass

    def get_preprocessor(self, hparams_seed):
        steps = self.get_preprocessing_steps()
        preprocessor = Pipeline(steps)
        params = self._get_preprocessor_params(hparams_seed)
        preprocessor.set_params(**params)
        return preprocessor

    @abstractmethod
    def load(self):
        pass
    
    def get_labels(self):
        _X, y, _groups = self.load()
        le = LabelEncoder().fit(y)
        return le.classes_

    def get_splits(self):
        X, y, groups = self.load()
        args = (X, y, groups, self.valid_size, self.test_size, self.seed)
        return _split_grouped_data(*args)


class RAData(Data):
    TREATMENT = 'therapy'
    GROUP = 'id'

    def get_column_transformer(self):
        # Numerical columns
        numerical_column_selector = make_column_selector(dtype_include='float64')
        numerical_column_pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            KBinsDiscretizer(subsample=None)  # Need a seed if `subsample != None`!
        )

        # Categorical columns.
        categorical_column_selector = make_column_selector(dtype_include='category')
        categorical_column_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(
                drop='if_binary',
                handle_unknown='ignore',
                sparse=False
            )
        )

        # Boolean columns.
        boolean_column_selector = make_column_selector(dtype_include='boolean')
        boolean_column_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent')
        )

        return ColumnTransformer(
            transformers=[
                ('numerical_transformer', numerical_column_pipeline, numerical_column_selector),
                ('categorical_transformer', categorical_column_pipeline, categorical_column_selector),
                ('boolean_transformer', boolean_column_pipeline, boolean_column_selector)
            ],
            remainder='passthrough'  # Passthrough the group column
        )

    def get_feature_selector(self):
        return None

    def get_preprocessing_steps(self):
        return (
            ('column_transformer', self.get_column_transformer()),
            ('feature_selector', self.get_feature_selector())
        )

    def load(self):
        data = pd.read_pickle(self.path)

        if self.sample_size is not None:
            np.random.seed(self.seed)
            sampled_groups = np.random.choice(
                data[self.GROUP], size=self.sample_size, replace=False
            )
            data = data.loc[data[self.GROUP].isin(sampled_groups)]

        X = data.drop(columns=[self.TREATMENT, self.GROUP])
        y = data[self.TREATMENT]
        groups = data[self.GROUP]

        return X, y, groups
