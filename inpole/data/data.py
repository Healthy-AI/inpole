import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import (
    make_column_transformer,
    make_column_selector,
    ColumnTransformer
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


def split_grouped_data(
    X, y, groups, valid_size, test_size, seed=None, return_label_encoder=False
):
    if isinstance(y, pd.Series):
        y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

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

    if return_label_encoder:
        return data_train, data_valid, data_test, le
    else:
        return data_train, data_valid, data_test


class SepsisData:
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
    TREATMENTS = [
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
        'max_dose_vaso',
        'input_4hourly_prev',
        'max_dose_vaso_prev'
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

    @staticmethod
    def get_scale_transform():
        return make_pipeline(
            StandardScaler()
        )

    @staticmethod
    def get_log_scale_transform():
        return make_pipeline(
            FunctionTransformer(
                lambda x: np.log(x + 0.1),
            ),
            StandardScaler()
        )

    @staticmethod
    def get_scale_columns(X):
        return sorted(list(set(X).intersection(SepsisData.SCALE)))

    @staticmethod
    def get_log_scale_columns(X):
        return sorted(list(set(X).intersection(SepsisData.LOG_SCALE)))

    @staticmethod
    def get_preprocessor():
        return make_column_transformer(
            (SepsisData.get_scale_transform(), SepsisData.get_scale_columns),
            (SepsisData.get_log_scale_transform(), SepsisData.get_log_scale_columns),
            remainder='passthrough'
        )

    @staticmethod
    def discretize_treatments(treatments):
        """Discretize treatments into 5 levels.

        Parameters
        ----------
        treatments : DataFrame of shape (n_samples,)
            Raw treatment data.

        Returns
        -------
        discrete_treatments : NumPy array of shape (n_samples, 2)
            Discrete treatments (values between 0 and 4).
        """
        discrete_treatments = np.zeros(treatments.size)  # 0 is default (zero dose)
        is_nonzero = treatments > 0
        ranked_nonzero_treatments = rankdata(treatments[is_nonzero]) / np.sum(is_nonzero)
        discrete_nonzero_treatments = np.digitize(ranked_nonzero_treatments, bins=[0., 0.25, 0.5, 0.75, 1.], right=True)
        discrete_treatments[is_nonzero] = discrete_nonzero_treatments
        return discrete_treatments

    @staticmethod
    def get_splits(data_path, *args, **kwargs):
        data = pd.read_csv(data_path)

        groups = data[SepsisData.GROUP]

        Y = data[SepsisData.TREATMENTS]
        Y_discrete = Y.apply(SepsisData.discretize_treatments, raw=True)
        _, y = np.unique(Y_discrete, axis=0, return_inverse=True)

        Yg = pd.concat([Y, groups], axis=1)
        previous_doses = Yg.groupby(by=SepsisData.GROUP).shift(1, fill_value=0)
        previous_doses = previous_doses.drop(SepsisData.GROUP, axis=1)
        rename = {c: c + '_prev' for c in previous_doses.columns}
        previous_doses = previous_doses.rename(rename, axis=1)

        X = data[SepsisData.FEATURES]
        X = pd.concat([X, previous_doses], axis=1)
        
        return split_grouped_data(X, y, groups, *args, **kwargs)


class AdniData:
    FEATURES = [
        'CDRSB_cat',
        'MRI_previous_outcome'
    ]
    TREATMENT = 'MRI_ordered'
    GROUP = 'RID'

    @staticmethod
    def get_column_transformer():
        return make_column_transformer(
            (OneHotEncoder(handle_unknown='error', sparse=False), AdniData.FEATURES),
            remainder='passthrough'  # Passthrough the group column
        )
    
    @staticmethod
    def get_feature_selector():
        return None
    
    @staticmethod
    def get_preprocessor():
        steps = (
            AdniData.get_column_transformer(),
            AdniData.get_feature_selector()
        )
        return make_pipeline(*steps)
    
    @staticmethod
    def get_splits(data_path, *args, **kwargs):
        data = pd.read_csv(data_path)
        
        X = data[AdniData.FEATURES]
        y = data[AdniData.TREATMENT]
        groups = data[AdniData.GROUP]

        return split_grouped_data(X, y, groups, *args, **kwargs)


class RAData:
    TREATMENT = 'therapy'
    GROUP = 'id'

    @staticmethod
    def get_column_transformer():
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
                ('categorical_transformer', categorical_column_pipeline, categorical_column_selector),
                ('boolean_transformer', boolean_column_pipeline, boolean_column_selector)
            ],
            remainder='passthrough'  # Passthrough the group column
        )

    @staticmethod
    def get_feature_selector():
        return None
    
    @staticmethod
    def get_preprocessor():
        steps = (
            RAData.get_column_transformer(),
            RAData.get_feature_selector()
        )
        return make_pipeline(*steps)

    @staticmethod
    def get_splits(data_path, *args, **kwargs):
        data = pd.read_pickle(data_path)

        np.random.seed(2023)
        sampled_groups = np.random.choice(data[RAData.GROUP], size=2000, replace=False)
        data = data.loc[data[RAData.GROUP].isin(sampled_groups)]

        X = data.drop(columns=[RAData.TREATMENT, RAData.GROUP])
        y = data[RAData.TREATMENT]
        groups = data[RAData.GROUP]

        return split_grouped_data(X, y, groups, *args, **kwargs)


class SynthData:
    FEATURES = [
        'age',
        'weight',
        'smoker',
    ]
    TREATMENT = 'MI'
    GROUP = 'id'

    @staticmethod
    def get_column_transformer():
        return make_column_transformer(
            (OneHotEncoder(handle_unknown='error', sparse=False), SynthData.FEATURES),
            remainder='passthrough'  # Passthrough the group column
        )

    @staticmethod
    def get_feature_selector():
        return None
    
    @staticmethod
    def get_preprocessor():
        steps = (
            SynthData.get_column_transformer(),
            SynthData.get_feature_selector()
        )
        return make_pipeline(*steps)

    @staticmethod
    def get_splits(data_path, *args, **kwargs):
        data = pd.read_csv(data_path)

        X = data[SynthData.FEATURES]
        y = data[SynthData.TREATMENT]
        groups = data[SynthData.GROUP]

        return split_grouped_data(X, y, groups, *args, **kwargs)
