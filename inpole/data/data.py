"""
Data handler for each experiment.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    LabelEncoder
)
from sklearn.compose import (
    make_column_selector,
    ColumnTransformer,
    make_column_transformer
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from . import hparam_registry


# Include new datasets here.
__all__ = [
    'get_data_handler_class',
    'get_data_handler_from_config',
    'RAData',
    'ADNIData',
    'SwitchData',
    'SepsisData'
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


def shift_variable(grouped, variable, period, fillna):
    shifted = grouped.shift(periods=period)
    shifted = shifted.fillna(fillna)
    shifted.rename(f'{variable}_{period}', inplace=True)
    return shifted


def discretize_doses(doses, num_levels):
    """Discretize continuous doses into `num_levels` levels.

    Parameters
    ----------
    doses : DataFrame of shape (n_samples,)
        Raw data.

    Returns
    -------
    discrete_doses : NumPy array of shape (n_samples,)
        Discrete doses (values between 0 and `num_levels-1`).
    """
    discrete_doses = np.zeros_like(doses)  # 0 is default (zero dose)
    is_nonzero = doses > 0
    ranked_nonzero_doses = rankdata(doses[is_nonzero]) / np.sum(is_nonzero)
    discrete_nonzero_doses = np.digitize(
        ranked_nonzero_doses,
        bins=np.linspace(0, 1, num=num_levels),
        right=True
    )
    discrete_doses[is_nonzero] = discrete_nonzero_doses
    return discrete_doses


def _add_log(x):
    return np.log(0.1 + x)


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
    def __init__(
        self,
        path,
        valid_size=0.2,
        test_size=0.2,
        seed=None,
        sample_size=None,
        shift_periods=0,
        shift_exclude=None,
    ):
        self.path = path
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed
        self.sample_size = sample_size
    
        if not isinstance(shift_periods, int):
            try:
                shift_periods = int(shift_periods)
            except TypeError:
                raise ValueError(
                    "`periods` must be an integer, "
                    f"got {type(shift_periods).__name__}."
                )
        self.shift_periods = shift_periods
        self.shift_exclude = shift_exclude
 
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

    def get_preprocessor(self, hparams_seed=None):
        steps = self.get_preprocessing_steps()
        preprocessor = Pipeline(steps)
        params = self._get_preprocessor_params(hparams_seed)
        preprocessor.set_params(**params)
        return preprocessor

    @abstractmethod
    def load(self):
        pass

    def sample(self, data):
        r = np.random.RandomState(self.seed)
        sampled_groups = r.choice(
            data[self.GROUP], size=self.sample_size, replace=False
        )
        return data.loc[data[self.GROUP].isin(sampled_groups)]
    
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

    THERAPIES = [
        'csdmard',
        'tnfi',
        'abatacept',
        'rituximab',
        'il-6',
        'jaki'
    ]

    def __init__(self, *, include_therapy_history=False, **kwargs):
        super().__init__(**kwargs)
        self.include_therapy_history = include_therapy_history

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

        if self.sample_size is not None:
            data = self.sample(data)
        
        is_registry_visit = ~data.visitdate.isna()

        X = data.drop(columns=[self.TREATMENT, self.GROUP, 'visitdate'])
        y = data[self.TREATMENT]
        groups = data[self.GROUP]

        if not self.include_therapy_history:
            X = X.drop(columns=map(lambda t: f'hx{t}', self.THERAPIES))

        if self.shift_periods > 0:
            registry_data = data.loc[is_registry_visit]
            grouped_registry_data = registry_data.groupby(groups)
            for c in data.columns:
                if c in self.shift_exclude:
                    continue
                if c == 'therapy':
                    grouped = data[c].groupby(groups)
                    fillna = data[c]
                else:
                    grouped = grouped_registry_data[c]
                    fillna = registry_data[c]
                for period in range(1, self.shift_periods + 1):
                    s = shift_variable(grouped, c, period, fillna)
                    s.reindex(data.index).to_frame()
                    X = pd.concat([X, s], axis=1)
                    fillna = s.squeeze()
        
        X = X.loc[is_registry_visit]
        y = y.loc[is_registry_visit]
        groups = groups.loc[is_registry_visit]

        return X, y, groups


class ADNIData(Data):
    TREATMENT = 'MRI_ordered'
    GROUP = 'RID'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column_transformer(self):
        # Numerical columns.
        numerical_column_selector = make_column_selector(dtype_include='float64')
        numerical_column_steps = [
            ('imputer', None),
            ('encoder', KBinsDiscretizer(subsample=None))
        ]
        numerical_column_pipeline = Pipeline(numerical_column_steps)

        # Categorical columns.
        categorical_column_selector = make_column_selector(dtype_include='category')
        categorical_column_steps = [
            ('imputer', None),
            ('encoder', OneHotEncoder(drop='if_binary', 
                                      handle_unknown='ignore', 
                                      sparse_output=False))
        ]
        categorical_column_pipeline = Pipeline(categorical_column_steps)

        return ColumnTransformer(
            transformers=[
                ('numerical_transformer', numerical_column_pipeline, numerical_column_selector),
                ('categorical_transformer', categorical_column_pipeline, categorical_column_selector),
            ],
            remainder='passthrough'  # Passthrough the group column
        )

    def get_feature_selector(self):
        return None
    
    def load(self):
        data = pd.read_csv(
            self.path,
            dtype={
                'RID': 'object',
                'CDRSB_cat': 'category',
                'MRI_previous_outcome': 'category',
                'MRI_ordered': 'int64',
                'AGE': 'float64',
                'PTGENDER': 'category',
                'PTMARRY': 'category',
                'PTEDUCAT': 'float64',
                'APOE4': 'category'
            }
        )

        X = data.drop(columns=[self.TREATMENT, self.GROUP])
        y = data[self.TREATMENT]
        groups = data[self.GROUP]

        if self.shift_periods > 0:
            grouped_data = data.groupby(groups)
            for c in data.columns:
                if c in self.shift_exclude:
                    continue
                grouped = grouped_data[c]
                fillna = data[c]
                for period in range(1, self.shift_periods + 1):
                    s = shift_variable(grouped, c, period, fillna)
                    X = pd.concat([X, s.to_frame()], axis=1)
                    fillna = s.squeeze()

        return X, y, groups


class SwitchData(RAData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load(self):
        X, y, groups = super().load()
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        y = y.groupby(groups).diff().fillna(0)
        y = (y != 0).astype(int)
        return X, y, groups


class SepsisData(Data):
    FEATURES = [
        'HR',
        'SysBP',
        'MeanBP',
        'DiaBP',
        'Shock_Index',
        'Hb',
        'BUN',
        'Creatinine',
        'output_4hourly',
        'Arterial_pH',
        'Arterial_BE',
        'HCO3',
        'Arterial_lactate',
        'PaO2_FiO2',
        'age',
        'elixhauser',
        'SOFA'
    ]
    TREATMENT = [
        'input_4hourly',
        'max_dose_vaso'
    ]
    GROUP = 'icustayid'

    LOG_SCALE = [
        'SpO2',
        'BUN',
        'Creatinine',
        'SGOT',
        'SGPT',
        'Total_bili',
        'INR',
        'input_total',
        'input_4hourly',
        'output_total',
        'output_4hourly',
        'max_dose_vaso'
    ]
    SCALE = [
        'age',
        'Weight_kg',
        'GCS',
        'HR',
        'SysBP',
        'MeanBP',
        'DiaBP',
        'RR',
        'Temp_C',
        'FiO2_1',
        'Potassium',
        'Sodium',
        'Chloride',
        'Glucose',
        'Magnesium',
        'Calcium',
        'Hb',
        'WBC_count',
        'Platelets_count',
        'PTT',
        'PT',
        'Arterial_pH',
        'paO2',
        'paCO2',
        'Arterial_BE',
        'HCO3',
        'Arterial_lactate',
        'SOFA',
        'SIRS',
        'Shock_Index',
        'PaO2_FiO2',
        'cumulated_balance',
        'elixhauser'
    ]

    def __init__(self, *, num_levels=5, **kwargs):
        self.num_levels = num_levels
        super().__init__(**kwargs)

    def get_scale_transformer(self):
        return make_pipeline(StandardScaler())

    def get_log_scale_transformer(self):
        return make_pipeline(
            FunctionTransformer(_add_log, feature_names_out='one-to-one'),
            StandardScaler()
        )

    def get_scaled_columns(self, X):
        scaled_columns = self.SCALE
        for period in range(1, self.shift_periods + 1):
            scaled_columns += [f'{c}_{period}' for c in self.SCALE]
        return sorted(list(set(X).intersection(scaled_columns)))

    def get_log_scaled_columns(self, X):
        log_scaled_columns = self.LOG_SCALE
        for period in range(1, self.shift_periods + 1):
            log_scaled_columns += [f'{c}_{period}' for c in self.LOG_SCALE]
        return sorted(list(set(X).intersection(log_scaled_columns)))

    def get_column_transformer(self):
        return make_column_transformer(
            (self.get_scale_transformer(), self.get_scaled_columns),
            (self.get_log_scale_transformer(), self.get_log_scaled_columns),
            remainder='passthrough'
        )

    def get_feature_selector(self):
            return None

    def load(self):
        data = pd.read_csv(self.path)

        if self.sample_size is not None:
            data = self.sample(data)

        X = data[self.FEATURES]
        groups = data[self.GROUP]

        Y = data[self.TREATMENT]
        Y_discrete = Y.apply(discretize_doses, raw=True, num_levels=self.num_levels)
        _, y = np.unique(Y_discrete, axis=0, return_inverse=True)

        if self.shift_periods > 0:
            grouped_data = data.groupby(groups)
            for c in data.columns:
                if c in self.shift_exclude:
                    continue
                grouped = grouped_data[c]
                fillna = data[c]
                for period in range(1, self.shift_periods + 1):
                    s = shift_variable(grouped, c, period, fillna)
                    X = pd.concat([X, s.to_frame()], axis=1)
                    fillna = s.squeeze()

        return X, y, groups
