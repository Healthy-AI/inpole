import os

import numpy as np
import pandas as pd


def make_adni(adni_path, out_path):
    """Preprocess the ADNI dataset as described in Pace et al. (2022)."""
    adni = pd.read_csv(adni_path, low_memory=False)
    
    # Filter out visits without a CDR-SB measurement.
    assert adni['CDRSB.bl'].notnull().all()
    has_cdrsb = adni.CDRSB.notnull()
    adni = adni[has_cdrsb]

    # Filter out visits without a diagnosis.
    has_dx = adni.DX.notnull()
    adni = adni[has_dx]

    # Filter out visits separated by more than 6 months from the previous one.
    months_between_visits = adni.groupby('RID').M.diff().fillna(0)
    is_regular_and_continuous = (months_between_visits <= 6.0).groupby(adni.RID).cummin()
    adni = adni[is_regular_and_continuous]

    # Filter out patients with less than 2 visits.
    visit_counts = adni.groupby('RID').size()
    patients_with_at_least_two_visits = visit_counts[visit_counts >= 2].index
    adni = adni[adni.RID.isin(patients_with_at_least_two_visits)]

    # Add action variable (MRI scan ordered).
    adni['MRI_ordered'] = adni.Hippocampus.notnull().astype(int)

    # Categorize CDR-SB measurements.
    adni['CDRSB_cat'] = pd.cut(
        adni.CDRSB,
        bins=[0, 0.5, 4.5, 18.5],
        right=False,
        labels=["CDR-SB normal", "CDR-SB questionable", "CDR-SB severe"]
    )

    # Categorize the MRI outcome of the previous visit.
    hippocampus_mean = adni.Hippocampus.mean(skipna=True)
    hippocampus_std = adni.Hippocampus.std(skipna=True)

    def mri_outcome(volume):
        if pd.isnull(volume):
            return "No MRI"
        if volume < hippocampus_mean - 0.5 * hippocampus_std:
            return "Vh low"
        elif volume > hippocampus_mean + 0.5 * hippocampus_std:
            return "Vh high"
        else:
            return "Vh average"

    adni["MRI_outcome"] = adni.Hippocampus.apply(mri_outcome)
    adni["MRI_previous_outcome"] = \
        adni.groupby('RID').MRI_outcome.shift(1, fill_value="No MRI")
    
    # Select features.
    dtypes = {
        'RID': 'object', 'CDRSB_cat': 'category',
        'MRI_previous_outcome': 'category', 'MRI_ordered': 'int64',
        'AGE': 'float64', 'PTGENDER': 'category', 'PTMARRY': 'category',
        'PTEDUCAT': 'float64', 'APOE4': 'category'
    }
    adni = adni[list(dtypes)]
    adni = adni.astype(dtypes)

    # Save the preprocessed dataset.
    file_name = os.path.join(out_path, 'adni.pkl')
    adni.to_pickle(file_name)


def get_NEWS2_score(respiratory_rate, SpO2, on_vent, blood_pressure, heart_rate, is_CVPU, temperature):
    """Calculate the National Early Warning 2 (NEWS2) score.
    
    Reference: https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2.
    """

    score_table = {
        'respiratory_rate': {
            'range': [0, 9, 12, 21, 25, float('inf')],
            'score': [3, 1, 0, 2, 3]
        },
        'SpO2_scale1': {
            'range': [0, 92, 94, 96, float('inf')],
            'score': [3, 2, 1, 0]
        },
        'SpO2_scale2': {
            'on_vent': {
                'range': [93, 95, 97, 100.01],
                'score': [4, 3, 2]
            },
            'on_air': {
                'range': [0, 84, 86, 88, 93],
                'score': [4, 3, 2, 0]
            }
        },
        'on_vent': {
            'range': [0, 1, float('inf')],
            'score': [2, 0]
        },
        'blood_pressure': {
            'range': [0, 90, 100, 110, 220, float('inf')],
            'score': [3, 2, 1, 0, 3]
        },
        'heart_rate': {
            'range': [0, 40, 50, 90, 110, 130, float('inf')],
            'score': [3, 1, 0, 1, 2, 3]
        },
        'is_CVPU': {
            'range': [0, 1, float('inf')],
            'score': [0, 3]
        },
        'temperature': {
            'range': [0, 35, 36, 38, 39, float('inf')],
            'score': [3, 1, 0, 1, 2]
        }
    }

    scoring_dict = {
        'respiratory_rate': respiratory_rate,
        'SpO2_scale1': SpO2,  # Assume no HRF
        'on_vent': on_vent,
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'is_CVPU': is_CVPU,
        'temperature': temperature
    }

    def get_single_score(selector, value):
        score_dict = score_table[selector]
        assert len(score_dict['range']) == len(score_dict['score']) + 1
        for idx, upper_bound in enumerate(score_dict['range']):
            if value < upper_bound:
                return score_dict['score'][idx - 1]
        raise ValueError("{} not in range of {}.".format(selector, score_dict['range']))
    
    total_score = 0
    for selector, value in scoring_dict.items():
        total_score += get_single_score(selector, value)
    return total_score


def calculate_news2(row):
    is_CVPU = 1 if row['GCS'] < 15 else 0
    return get_NEWS2_score(
        respiratory_rate=row['RR'],
        SpO2=row['SpO2'],
        on_vent=row['mechvent'],
        blood_pressure=row['SysBP'],
        heart_rate=row['HR'],
        is_CVPU=is_CVPU,
        temperature=row['Temp_C'],
    )


def make_sepsis(sepsis_path, out_path, exclude_patients_with_missing_data=False):
    sepsis = pd.read_csv(sepsis_path, low_memory=False)

    if exclude_patients_with_missing_data:
        sepsis['step'] = sepsis['bloc'].astype(int) - 1
        max_steps = sepsis.groupby('icustayid').step.max()
        actual_counts = sepsis.groupby('icustayid').size()
        expected_counts = max_steps + 1
        discontinuous_patients = expected_counts[expected_counts != actual_counts].index
        sepsis = sepsis[~sepsis.icustayid.isin(discontinuous_patients)]
        sepsis.sort_values(['id', 'step'], ascending=True, inplace=True)

        # Filter out patients with less than 20--24 hours of data.
        #
        # 0--4: 0 | 4--8: 1 | 8--12: 2 | 12--16: 3 | 16--20: 4 | 20--24: 5
        p_gt24 = sepsis.groupby('icustayid').step.max() >= 5
        sepsis = sepsis[
            sepsis.icustayid.isin([p for p, true in p_gt24.items() if true])
        ]
        assert (sepsis.groupby('icustayid').tail(1).index.get_level_values('step').to_numpy() >= 5).all()

        max_len = 17  # 72 hours
        sepsis = sepsis.groupby('icustayid').head(max_len + 1)
        assert sepsis.step.max() == max_len

        sepsis.reset_index(drop=True, inplace=True)

    # Add NEWS2 score.
    sepsis['NEWS2'] = sepsis.apply(calculate_news2, axis=1)

    # Save the preprocessed dataset.
    file_name = os.path.join(out_path, 'sepsis.pkl')
    sepsis.to_pickle(file_name)


class COPDPreprocessing:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.continuous_columns = []

    def create_treatment_columns(self):
        # Fluid intakes
        conditions = [(self.df[col] != 0) for col in ['MEDS_220949.0', 'MEDS_225943.0', 'MEDS_225158.0']]
        self.df['Fluids_intakes'] = np.select(conditions, [self.df[col] for col in ['MEDS_220949.0', 'MEDS_225943.0', 'MEDS_225158.0']], default=0)
        self.df.drop(['MEDS_220949.0', 'MEDS_225943.0', 'MEDS_225158.0'], axis=1, inplace=True)

        # Sedation
        self.df['sedation'] = np.where(self.df['MEDS_222168.0'] != 0, self.df['MEDS_222168.0'], self.df['MEDS_225942.0'])
        self.df.drop(['MEDS_222168.0', 'MEDS_225942.0'], axis=1, inplace=True)

    def limit_missingness(self):
        percent_missing = self.df.isna().mean() * 100
        exclude_columns = ['sedation', 'Fluids_intakes']
        columns_to_drop = percent_missing[(percent_missing > 20) & (~percent_missing.index.isin(exclude_columns))].index
        self.df = self.df.drop(columns=columns_to_drop)

    def impute_missing_values(self):
        # Impute continuous columns with mean per patient
        self.continuous_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns if col not in ['PatientID']]
        self.df[self.continuous_columns] = self.df.groupby('PatientID')[self.continuous_columns].transform(lambda x: x.fillna(x.mean()))

        # Fill remaining NaNs with zero
        self.df.fillna(0, inplace=True)

    def save_data(self, filename):
        self.df.to_pickle(filename)


def make_copd(copd_path, out_path):
    copd = COPDPreprocessing(copd_path)
    copd.create_treatment_columns()
    copd.limit_missingness()
    copd.impute_missing_values()
    copd.save_data(os.path.join(out_path, 'copd.pkl'))
