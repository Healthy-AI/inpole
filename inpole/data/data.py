"""
Data handler for each experiment.
"""

from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    LabelEncoder
)
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import (
    make_column_selector,
    ColumnTransformer,
    make_column_transformer
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier

from . import hparam_registry
from .utils import shift_variable


# Include new datasets here.
__all__ = [
    'get_data_handler_class',
    'get_data_handler_from_config',
    'RAData',
    'ADNIData',
    'SwitchData',
    'SepsisData',
    'COPDData'
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


def discretize_doses(doses, num_levels):
    """Discretize continuous treatment doses into `num_levels` levels.

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


class FunctionTransformer(preprocessing.FunctionTransformer):
    def fit(self, X, y, agg_index=None):
        if self.kw_args is None:
            self.kw_args = {'agg_index': agg_index}
        else:
            self.kw_args['agg_index'] = agg_index
        return super().fit(X, y)


class Data(ABC):
    def __init__(
        self,
        path,
        valid_size=0.2,
        test_size=0.2,
        seed=None,
        sample_size=None,
        include_context_variables=True,
        include_previous_treatment=True,
        fillna_value=None,
        aggregate_history=False,
        add_current_context=False,
        aggregate_exclude=None,
        shift_periods=0,
        shift_exclude=None,
        max_features=None,
    ):
        self.path = path
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed
        self.sample_size = sample_size
        self.include_context_variables = include_context_variables
        self.include_previous_treatment = include_previous_treatment
        self.fillna_value = fillna_value
        self.aggregate_history = aggregate_history
        self.add_current_context = add_current_context
        self.aggregate_exclude = aggregate_exclude
        self.shift_periods = shift_periods
        self.shift_exclude = shift_exclude
        self.max_features = max_features

        self.validate_arguments()
    
    def validate_arguments(self):
        if not isinstance(self.shift_periods, int):
            try:
                self.shift_periods = int(self.shift_periods)
            except TypeError:
                raise ValueError(
                    "`periods` must be an integer, "
                    f"got {type(self.shift_periods).__name__}."
                )

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
    
    def get_preprocessing_steps(self, cont_feat_trans):
        # @TODO: Rename the second step (feature_selector).
        return (
            ('column_transformer', self.get_column_transformer(cont_feat_trans)),
            ('feature_selector', self.get_feature_selector())
        )

    @abstractmethod
    def get_column_transformer(self, cont_feat_trans):
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

    def _aggregate_history(self, X, agg_index):
        def func(x):
            return x.max() if x.name in X.columns[agg_index] else x.iloc[-1]
        X = pd.DataFrame(X)
        c_group = X.columns[-1]
        aggregates = []
        for _, sequence in X.groupby(by=c_group, sort=False):
            sequence = sequence.drop(columns=c_group)
            t = 0
            while t < len(sequence):
                aggregates += [sequence.iloc[:t+1].apply(func)]
                t += 1
        return np.array(aggregates)
    
    def _add_columns_to_aggregate(self, X):
        mapper = {
            c: f'{c}_agg' for c in X.columns
            if c not in self.aggregate_exclude
        }
        X_agg = X.rename(mapper, axis=1)
        if self.add_current_context:
            X_add = X.drop(columns=self.aggregate_exclude)
            X_agg = pd.concat([X_agg, X_add], axis=1)
        return X_agg

    def _get_feature_names_out(self, _, input_features=None):
        return [x for x in input_features if not x.startswith('remainder')]

    def get_feature_selector(self):
        steps = []
        if self.aggregate_history:
            aggregator = FunctionTransformer(self._aggregate_history,
                                             feature_names_out=self._get_feature_names_out)
        else:
            aggregator = None
        if self.max_features is None:
            selector = None
        else:
            # @TODO: Should not have a fixed seed here.
            selector = SelectFromModel(
                estimator=ExtraTreesClassifier(
                    n_estimators=50,
                    random_state=2024
                ),
                threshold=-np.inf,
                max_features=self.max_features
            )
        steps = [('aggregator', aggregator), ('selector', selector)]
        return Pipeline(steps)

    def get_preprocessor(self, cont_feat_trans, hparams_seed=None):
        steps = self.get_preprocessing_steps(cont_feat_trans)
        preprocessor = Pipeline(steps)
        params = self._get_preprocessor_params(hparams_seed)
        preprocessor.set_params(**params)
        return preprocessor

    def _add_previous_treatment(self, X, data):
        grouped = data[self.TREATMENT].groupby(data[self.GROUP])
        previous_treatment = grouped.shift()
        if isinstance(self.TREATMENT, list):
            mapper = {c: 'prev_' + c for c in self.TREATMENT}
            previous_treatment.rename(columns=mapper, inplace=True)
            if not isinstance(self.fillna_value, list):
                fillna_value = [self.fillna_value] * len(self.TREATMENT)
            else:
                fillna_value = self.fillna_value
            assert len(fillna_value) == len(self.TREATMENT)
            fillna_value = dict(zip(mapper.values(), fillna_value))
            # @TODO: Handle the case when `previous_treatment` is a pandas Categorical.
            previous_treatment.apply('fillna', value=fillna_value, inplace=True)
        else:
            previous_treatment.rename('prev_' + self.TREATMENT, inplace=True)
            if not self.fillna_value in previous_treatment.cat.categories:
                previous_treatment = previous_treatment.cat.add_categories(self.fillna_value)
            previous_treatment.fillna(self.fillna_value, inplace=True)
        X = pd.concat([X, previous_treatment], axis=1)
        return X

    def _remove_previous_treatment(self, X):
        return X

    def _manipulate(self, X, y, groups, data):
        return X, y, groups

    def load(self):
        if self.path.endswith('.csv'):
            data = pd.read_csv(self.path)
        elif self.path.endswith('.pkl'):
            data = pd.read_pickle(self.path)
        else:
            file_format = self.path.split('.')[-1]
            raise ValueError(f"Unknown file format: {file_format}.")

        if self.sample_size is not None:
            data = self._sample(data)

        if not self.include_context_variables:
            X = pd.DataFrame(index=data.index)
        elif hasattr(self, 'FEATURES'):
            X = data[self.FEATURES]
        else:
            columns = self.TREATMENT + [self.GROUP] \
                if isinstance(self.TREATMENT, list) else [self.TREATMENT, self.GROUP]
            X = data.drop(columns=columns)

        assert not any(['agg' in c for c in X.columns])
        
        if isinstance(self.TREATMENT, list):
            Y = data[self.TREATMENT]
            Y_discrete = Y.apply(discretize_doses, raw=True,
                                 num_levels=self.num_levels)
            _, y = np.unique(Y_discrete, axis=0, return_inverse=True)
        else:
            y = data[self.TREATMENT]
        
        groups = data[self.GROUP]

        if self.include_previous_treatment:
            X = self._add_previous_treatment(X, data)
        else:
            X = self._remove_previous_treatment(X)
        
        if self.aggregate_history:
            X = self._add_columns_to_aggregate(X)
        
        X, y, groups = self._manipulate(X, y, groups, data)

        if self.shift_periods > 0:
            X = self._add_shifted_inputs(X, groups)
        
        return X, y, groups

    def _add_shifted_inputs(self, X, groups):
        grouped = X.groupby(groups)
        for c in X.columns:
            if c in self.shift_exclude or c.endswith('_agg'):
                continue
            fillna = X[c]
            for period in range(1, self.shift_periods + 1):
                s = shift_variable(grouped[c], period, fillna)
                s.reindex(X.index).to_frame()
                X = pd.concat([X, s], axis=1)
                fillna = s.squeeze()
        return X

    def _sample(self, data):
        r = np.random.RandomState(self.seed)
        sampled_groups = r.choice(
            data[self.GROUP], size=self.sample_size, replace=False
        )
        return data.loc[data[self.GROUP].isin(sampled_groups)]
    
    def get_labels(self):
        _X, y, _groups = self.load()
        le = LabelEncoder().fit(y)
        return le.classes_

    def _split_grouped_data(self, X, y, groups):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        y = LabelEncoder().fit_transform(y)

        Xg = pd.concat([X, groups], axis=1)

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, 
                                random_state=self.seed)
        ii_train, ii_test = next(gss.split(X, y, groups))
        
        Xg_train, y_train = Xg.iloc[ii_train], y[ii_train]
        Xg_test, y_test = Xg.iloc[ii_test], y[ii_test]
        groups_train = groups.iloc[ii_train]

        if self.valid_size > 0:
            train_size = 1 - self.test_size
            _valid_size = self.valid_size / train_size
            
            gss = GroupShuffleSplit(n_splits=1, test_size=_valid_size, 
                                    random_state=self.seed)
            ii_train, ii_valid = next(gss.split(Xg_train, y_train, groups_train))
            
            Xg_valid, y_valid = Xg_train.iloc[ii_valid], y_train[ii_valid]
            Xg_train, y_train = Xg_train.iloc[ii_train], y_train[ii_train]
        else:
            Xg_valid, y_valid = None, None

        data_train = (Xg_train, y_train)
        data_valid = (Xg_valid, y_valid)
        data_test = (Xg_test, y_test)

        return data_train, data_valid, data_test

    def get_splits(self):
        X, y, groups = self.load()
        return self._split_grouped_data(X, y, groups)


class RAData(Data):
    TREATMENT = 'therapy'
    GROUP = 'id'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column_transformer(self, cont_feat_trans):
        # Numerical columns.
        numerical_column_selector = make_column_selector(dtype_include='float64')
        numerical_column_steps = [('imputer', SimpleImputer(strategy='mean'))]
        if cont_feat_trans == 'discretize':
            numerical_column_steps += [
                ('encoder', KBinsDiscretizer(subsample=None))
            ]
        elif cont_feat_trans == 'scale':
            numerical_column_steps += [('encoder', StandardScaler())]
        else:
            numerical_column_steps += [('encoder', None)]
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
            remainder='passthrough'
        )

    def _manipulate(self, X, y, groups, data):
        is_registry_visit = ~data.visitdate.isna()
        X = X.loc[is_registry_visit]
        if 'visitdate' in X.columns:
            X = X.drop(columns='visitdate')
        y = y.loc[is_registry_visit]
        groups = groups.loc[is_registry_visit]
        return X, y, groups


class ADNIData(Data):
    TREATMENT = 'MRI_ordered'
    GROUP = 'RID'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_column_transformer(self, cont_feat_trans):
        # Numerical columns.
        numerical_column_selector = make_column_selector(dtype_include='float64')
        numerical_column_steps = [('imputer', None)]
        if cont_feat_trans == 'discretize':
            numerical_column_steps += [
                ('encoder', KBinsDiscretizer(subsample=None))
            ]
        elif cont_feat_trans == 'scale':
            numerical_column_steps += [('encoder', StandardScaler())]
        else:
            numerical_column_steps += [('encoder', None)]
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
            remainder='passthrough'
        )

    def _add_previous_treatment(self, X, data):
        if not self.include_context_variables:
            X = data['MRI_previous_outcome']
        return X
    
    def _remove_previous_treatment(self, X):
        return X.drop(columns='MRI_previous_outcome')


class SwitchData(RAData):
    def __init__(self, *, include_therapies=False, **kwargs):
        self.include_therapies = include_therapies
        super().__init__(**kwargs)

    def _add_previous_treatment(self, X, data):
        y = data[self.TREATMENT]
        groups = data[self.GROUP]
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        y = y.groupby(groups).diff()
        y = y.where(y.isna(), y != 0)  # Keep the NaNs
        previous_action = y.groupby(groups).shift()
        previous_action.replace({True: 'switch', False: 'stay'}, inplace=True)
        previous_action.fillna(self.fillna_value, inplace=True)
        previous_action.rename('prev_action', inplace=True)
        previous_action = previous_action.astype('category')
        X = pd.concat([X, previous_action], axis=1)
        if self.include_therapies:
            return super()._add_previous_treatment(X, data)
        else:
            return X
    
    def load(self):
        X, y, groups = super().load()
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        y = y.groupby(groups).diff()
        # Remove the first visit for each patient since we need two
        # consecutive visits to determine the action.
        X = X[~y.isna()]
        groups = groups[~y.isna()]
        y = y.dropna()
        y = (y != 0).astype(int)
        return X, y, groups


class SepsisData(Data):
    TREATMENT = [
        'input_4hourly',
        'max_dose_vaso'
    ]
    GROUP = 'icustayid'

    FEATURES = [
        'gender',
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

    SHIFT = [
        'gender',
        'mechvent',
        're_admission'
    ]
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
        'max_dose_vaso',
        'prev_input_4hourly',
        'prev_max_dose_vaso'
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

    def get_shift_transformer(self):
        return make_pipeline(MinMaxScaler((-0.5, 0.5)))

    def get_scale_transformer(self):
        return make_pipeline(StandardScaler())

    def _add_log(self, x):
        return np.log(0.1 + x)

    def get_log_scale_transformer(self):
        return make_pipeline(
            preprocessing.FunctionTransformer(self._add_log, 
                                              feature_names_out='one-to-one'),
            StandardScaler()
        )

    def get_shifted_columns(self, X):
        shifted_columns = deepcopy(self.SHIFT)
        for period in range(1, self.shift_periods + 1):
            shifted_columns += [f'{c}_{period}' for c in self.SHIFT]
        if self.aggregate_history:
            shifted_columns += [f'{c}_agg' for c in self.SHIFT]
        return sorted(list(set(X).intersection(shifted_columns)))

    def get_scaled_columns(self, X):
        scaled_columns = deepcopy(self.SCALE)
        for period in range(1, self.shift_periods + 1):
            scaled_columns += [f'{c}_{period}' for c in self.SCALE]
        if self.aggregate_history:
            scaled_columns += [f'{c}_agg' for c in self.SCALE]
        return sorted(list(set(X).intersection(scaled_columns)))

    def get_log_scaled_columns(self, X):
        log_scaled_columns = deepcopy(self.LOG_SCALE)
        for period in range(1, self.shift_periods + 1):
            log_scaled_columns += [f'{c}_{period}' for c in self.LOG_SCALE]
        if self.aggregate_history:
            log_scaled_columns += [f'{c}_agg' for c in self.LOG_SCALE]
        return sorted(list(set(X).intersection(log_scaled_columns)))

    def get_column_transformer(self, cont_feat_trans):
        return make_column_transformer(
            (self.get_shift_transformer(), self.get_shifted_columns),
            (self.get_scale_transformer(), self.get_scaled_columns),
            (self.get_log_scale_transformer(), self.get_log_scaled_columns),
            remainder='passthrough'
        )


class COPDData(Data):
    TREATMENT = [
        'Fluids_intakes',
        'sedation'
    ]
    GROUP = 'PatientID'

    def __init__(self, *, num_levels=5, **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels

    def get_column_transformer(self, cont_feat_trans):
        categorical_columns = make_column_selector(dtype_include='object')
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        continuous_columns = make_column_selector(dtype_include='float64')
        continuous_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]
        )

        return make_column_transformer(
            (categorical_transformer, categorical_columns),
            (continuous_transformer, continuous_columns),
            remainder='passthrough'
        )
