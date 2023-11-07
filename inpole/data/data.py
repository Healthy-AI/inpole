"""
Data handler for each experiment.
"""

import re
from functools import partial
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
    make_column_transformer,
    make_column_selector,
    ColumnTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from . import hparam_registry


# Include new datasets here.
__all__ = [
    'get_data_handler_class',
    'get_data_handler_from_config',
    'RAData',
    'ADNIData',
    'SwitchData'
]


def get_data_handler_class(name):
    return globals()[name]


def get_data_handler_from_config(config):
    experiment = config['experiment']
    try:
        data_handler_name = experiment.upper() + 'Data'
        data_handler_class = get_data_handler_class(data_handler_name)
    except KeyError:
        try:
            data_handler_name = experiment.capitalize() + 'Data'
            data_handler_class = get_data_handler_class(data_handler_name)
        except KeyError:
            raise ValueError(f"Unknown experiment: {experiment}.")
    return data_handler_class(**config['data'])


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
    def ALIAS(self):
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
            hparams = hparam_registry.default_hparams(self.ALIAS)
        else:
            hparams = hparam_registry.random_hparams(self.ALIAS, hparams_seed)
        return hparams
    
    def get_preprocessing_steps(self):
        return (
            ('column_transformer', self.get_column_transformer()),
            ('feature_selector', self.get_feature_selector())
        )

    @abstractmethod
    def get_column_transformer(self):
        """Get the `ColumnTransformer` object that should be applied to the
        data.
        
        Each transformer should be a pipeline with the following steps:
        * 'imputer' (e.g., `SimpleImputer`)
        * 'encoder' (e.g., `OneHotEncoder`).
        
        The name of the transformers is not important, i.e., the 
        `ColumnTransformer` object may be created using the function
        `make_column_transformer`.
        """
        pass

    @abstractmethod
    def get_feature_selector(self):
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

    def __init__(self, *, shift=None, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift

    def get_column_transformer(self):
        # Numerical columns.
        numerical_column_selector = make_column_selector(dtype_include='float64')
        numerical_column_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('encoder', KBinsDiscretizer(subsample=None))  # Pass a seed if `subsample != None`!
        ]
        numerical_column_pipeline = Pipeline(numerical_column_steps)

        # Categorical columns.
        categorical_column_selector = make_column_selector(dtype_include='category')
        categorical_column_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='if_binary', 
                                      handle_unknown='ignore', 
                                      sparse_output=False))
        ]
        categorical_column_pipeline = Pipeline(categorical_column_steps)

        # Boolean columns.
        boolean_column_selector = make_column_selector(dtype_include='boolean')
        boolean_column_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', None)
        ]
        boolean_column_pipeline = Pipeline(boolean_column_steps)

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

    def load(self):
        data = pd.read_pickle(self.path)

        # Drop all columns with shifted values.
        match = partial(re.match, r'(.+)_([0-9]+)')
        c_shifted = [
            c for c in data.columns
            if match(c) and match(c).group(1) in data.columns
        ]
        shifted = data[c_shifted]
        data.drop(columns=c_shifted, inplace=True)

        if self.shift is not None:
            # Include the shifted columns that were specified in the
            # configuration file.
            shifted_variables = [match(c).group(1) for c in c_shifted]
            for v in self.shift:
                if not v in shifted_variables:
                    raise ValueError(f"Column '{v}' is not shifted in the data.")
            shifted = shifted[
                [c for c in c_shifted if match(c).group(1) in self.shift]
            ]
            data = pd.concat([data, shifted], axis=1)

        if self.sample_size is not None:
            r = np.random.RandomState(self.seed)
            sampled_groups = r.choice(
                data[self.GROUP], size=self.sample_size, replace=False
            )
            data = data.loc[data[self.GROUP].isin(sampled_groups)]

        X = data.drop(columns=[self.TREATMENT, self.GROUP])
        y = data[self.TREATMENT]
        groups = data[self.GROUP]

        return X, y, groups


class ADNIData(Data):
    FEATURES = [
        'CDRSB_cat',
        'MRI_previous_outcome'
    ]
    TREATMENT = 'MRI_ordered'
    GROUP = 'RID'

    def get_column_transformer(self):
        steps = [
            ('imputer', None),
            ('encoder', OneHotEncoder(handle_unknown='error', 
                                      sparse_output=False))
        ]
        return make_column_transformer(
            (Pipeline(steps), self.FEATURES),
            remainder='passthrough'  # Passthrough the group column
        )

    def get_feature_selector(self):
        return None
    
    def load(self):
        data = pd.read_csv(self.path)

        X = data[self.FEATURES]
        y = data[self.TREATMENT]
        groups = data[self.GROUP]

        return X, y, groups


class SwitchData(RAData):
    def __init__(self, *, shift=None, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
    
    def load(self):
        X, y, groups = super().load()
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        y = y.groupby(groups, group_keys=False).apply(pd.Series.diff).fillna(0)
        y = (y != 0).astype(int)
        return X, y, groups
