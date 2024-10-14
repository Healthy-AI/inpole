import os
import warnings
import itertools

import numpy as np
import pandas as pd

try:
    from .corevitas import COREVITAS_DATA
except ImportError:
    import os
    import pickle
    if not os.environ.get("COREVITAS_DATA_PATH"):
        raise ValueError(
            "The environment variable 'COREVITAS_DATA_PATH' is not set."
        )
    with open(os.environ("COREVITAS_DATA_PATH"), "rb") as f:
        COREVITAS_DATA = pickle.load(f)


# =============================================================================
# == ADNI =====================================================================
# =============================================================================

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


# =============================================================================
# == RA =======================================================================
# =============================================================================

def get_all_variables():
    all_variables = {}
    drugs = itertools.chain.from_iterable(COREVITAS_DATA.drug_classes.values())
    for v, i in COREVITAS_DATA.variables.items():
        if "%s" in v:
            for d in drugs:
                all_variables[v % d] = i
        else:
            all_variables[v] = i
    return all_variables


def categorize_bp(row):
    # https://www.heart.org/en/health-topics/high-blood-pressure
    if row.seatedbp1 > 180 or row.seatedbp2 > 120:
        return 'htn stage 3'
    elif row.seatedbp1 >= 140 or row.seatedbp2 >= 90:
        return 'htn stage 2'
    elif row.seatedbp1 >= 130 or row.seatedbp2 >= 80:
        return 'htn stage 1'
    elif 120 <= row.seatedbp1 < 130 and row.seatedbp2 < 80:
        return 'elevated'
    elif row.seatedbp1 < 120 and row.seatedbp2 < 80:
        return 'normal'
    else:
        return np.nan


def summarize(X, name, impute=False):
    if impute:
        return pd.Series(X.sum(axis=1) > 0, name=name, dtype='boolean')
    else:
        X = X.sum(axis=1)
        X = X.where(X.isna(), X > 0)
        X.name = name
        return X.astype('boolean')


def assign_stage(X):
    """Assign the stage of treatment."""

    def update_stage(x):
        try:
            i = x[x == 1].index[0]
        except IndexError:
            return x
        j = x.index[-1]
        x.loc[i:j] = range(1, 2 + j-i)
        return x

    if 'stage' in X.columns:
        X = X.drop(columns='stage')

    X = X.assign(stage=-1)

    is_baseline = (X.hxbiojak == 0) & (X.initbiojak == 1)
    X.loc[is_baseline, 'stage'] = 1

    X['stage'] = X.groupby('id', group_keys=False).stage.apply(update_stage)

    return X.astype({'stage': 'float64'})


def require_valid_therapies(x):
    if not x.stage.eq(1).any():
        return x
    idx_first_invalid = x[
        x.stage.ge(1) & x.therapy.eq("Other therapy")
    ].first_valid_index()
    return x if idx_first_invalid is None else x.loc[:idx_first_invalid-1]


def make_ra(ra_path, out_path, impute=True):
    # Load the preprocessed RA dataset.
    ra = pd.read_pickle(ra_path)

    # Assign the stage of treatment.
    ra = pd.concat(
        [
            assign_stage(ra[ra.visitdate.notna()]),
            ra[ra.visitdate.isna()]
        ]
    ).sort_values(by=['index', 'date'])

    # Filter out patients without any index visit.
    eligible_ids = ra[ra.stage.eq(1)].id.unique()
    ra = ra[ra.id.isin(eligible_ids)]

    # Filter out patients with invalid therapies.
    ra = ra.groupby('id', group_keys=False).apply(require_valid_therapies)

    # Filter out patients with no follow-up visits.
    has_followup = ra[ra.stage.ge(1)].groupby('id').size() > 1
    eligible_ids = has_followup[has_followup].index
    ra = ra[ra.id.isin(eligible_ids)]

    # Decode categorical variables.
    for v, i in get_all_variables().items():
        if v in ra.columns and 'categories' in i:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                ra.loc[:, v] = ra[v].map(i['categories'])

    # Categorize BMI.
    bins = [0, 18.4, 24.9, 29.9, np.inf]
    ra['bmi_cat'] = pd.cut(
        ra.bmi, bins=bins,
        labels=['underweight', 'healthy', 'overweight', 'obesity'],
        include_lowest=True
    )

    # Categorize CDAI.
    bins = [0, 2.8, 10, 22, np.inf]
    ra['cdai_cat'] = pd.cut(
        ra.cdai, bins=bins, labels=['remission', 'low', 'moderate', 'high'],
        include_lowest=True
    )

    # Categorize blood pressure.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        ra['bp_cat'] = ra.apply(categorize_bp, axis=1).astype('category')

    # Include variables that should be imputed using forward filling (if `impute=True`).
    c_ffill = [
        'age', 'gender', 'college_completed', 'work_status', 'bmi_cat',
        'bp_cat', 'smoker', 'drinker', 'insurance_private', 'insurance_medicare',
        'insurance_medicaid', 'insurance_none', 'cdai_cat', 'duration_ra',
    ]
    X_ffill = ra[c_ffill]
    if impute:
        X_ffill = X_ffill.groupby(by=ra.id).fillna(method='ffill')

    # Include indicators of pregnancy.
    X_pregnancy = ra[['pregnant_since', 'pregnant_current']]

    # Include indicators of RF and CCP positivity.
    ccpos = ra.ccppos.astype('string')
    rfpos = ra.rfpos.astype('string')
    if impute:
        ccpos = ccpos.fillna('Not tested')
        rfpos = rfpos.fillna('Not tested')
    to_replace = {'False': 'Negative', 'True': 'Positive'}
    ccpos = ccpos.replace(to_replace).astype('category')
    rfpos = rfpos.replace(to_replace).astype('category')
    X_ra_biomarkers = pd.concat([ccpos, rfpos], axis=1)
    X_ra_biomarkers = X_ra_biomarkers.rename(columns={'ccppos': 'ccp', 'rfpos': 'rf'})

    # Include TB test results.
    tb = ra.ppd.astype('string')
    if impute:
        tb = tb.fillna('Not tested')
    to_replace = {'False': 'Negative', 'True': 'Positive'}
    X_tb = tb.replace(to_replace).astype('category')

    # Include variables indicating joint damage.
    X_joints = ra[['erosdis', 'jtspnarrow', 'jtdeform']]

    # Include variables indicating severe infections.
    infections = ['hospinf', 'ivinf']
    X_infections = summarize(ra[infections], name='infections', impute=impute)

    # Include variables indicating comorbidities.

    comor_metabolic = ['comor_hld', 'comor_diabetes']
    X_comor_metabolic = summarize(ra[comor_metabolic], 'comor_metabolic', impute)

    comor_cvd = [
        'comor_htn_hosp', 'comor_htn', 'comor_revasc', 'comor_ven_arrhythm',
        'comor_mi', 'comor_acs', 'comor_unstab_ang', 'comor_cor_art_dis',
        'comor_chf_hosp', 'comor_chf_nohosp', 'comor_stroke', 'comor_tia',
        'comor_card_arrest', 'comor_oth_clot', 'comor_pulm_emb',
        'comor_pef_art_dis', 'comor_pat_event', 'comor_urg_par', 'comor_pi',
        'comor_carotid', 'comor_other_cv',
    ]
    X_comor_cvd = summarize(ra[comor_cvd], 'comor_cvd', impute)

    comor_respiratory = ['comor_copd', 'comor_asthma', 'comor_fib']
    X_comor_respiratory = summarize(ra[comor_respiratory], 'comor_respiratory', impute)

    comor_dil = ['comor_drug_ind_sle']
    X_comor_dil = summarize(ra[comor_dil], 'comor_dil', impute)

    comor_cancer = [
        'comor_bc', 'comor_lc', 'comor_lymphoma', 'comor_skin_cancer_squa',
        'comor_skin_cancer_mel', 'comor_oth_cancer',
    ]
    X_comor_cancer = summarize(ra[comor_cancer], 'comor_cancer', impute)

    comor_gi_liver = [
        'comor_ulcer', 'comor_bowel_perf', 'comor_hepatic_wbiop',
        'comor_hepatic_nobiop',
    ]
    X_comor_gi_liver = summarize(ra[comor_gi_liver], 'comor_gi_liver', impute)

    comor_musculoskeletal = ['sec_sjog', 'jt_deform']
    to_replace = {'no': False, 'yes': True, 'new': True}
    X_comor_musculoskeletal = summarize(
        ra[comor_musculoskeletal].replace(to_replace).astype('boolean'),
        'comor_musculoskeletal',
        impute,
    )

    comor_other = [
        'comor_psoriasis', 'comor_depression', 'comor_fm', 'comor_oth_neuro',
        'comor_hemorg_hosp', 'comor_hemorg_nohosp', 'comor_oth_cond',
    ]
    X_comor_other = summarize(ra[comor_other], 'comor_other', impute)

    X_comor = [
        X_comor_metabolic, X_comor_cvd, X_comor_respiratory, X_comor_dil,
        X_comor_cancer, X_comor_gi_liver, X_comor_musculoskeletal, X_comor_other
    ]
    X_comor = pd.concat(X_comor, axis=1)

    # Combine all variables.
    X = [
        ra.date.dt.year.rename('year').astype('float64'), X_ffill, X_pregnancy,
        X_ra_biomarkers, X_tb, X_joints, X_infections, X_comor, ra.stage,
    ]
    X = pd.concat(X, axis=1)

    # Replace prebaseline therapies classified as "Other therapy" with
    # "csDMARD therapy" to avoid introducing an unused category in the data
    # pipeline.
    assert not 'Other therapy' in ra.loc[ra.stage.ge(1), 'therapy']
    ra.loc[ra.therapy.eq('Other therapy'), 'therapy'] = 'csDMARD therapy'

    # Save the data.
    Xgy = pd.concat([X, ra.therapy.astype('category'), ra.id], axis=1)
    Xgy.to_pickle(os.path.join(out_path, 'ra.pkl'))


# =============================================================================
# == Sepsis ===================================================================
# =============================================================================

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


# =============================================================================
# == COPD =====================================================================
# =============================================================================

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
