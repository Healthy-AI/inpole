from .models import *
from .data import *

# Each experiment maps to a dataset.
EXPERIMENTS = {
    'ra': RAData,
    'adni': ADNIData,
    'switch': SwitchData,
    'sepsis': SepsisData
}

NET_ESTIMATORS = {
    'sdt': SDTClassifer,
    'pronet': ProNetClassifier,
    'mlp': MLPClassifier
}

RECURRENT_NET_ESTIMATORS = {
    'rdt': RDTClassifer,
    'prosenet': ProSeNetClassifier,
    'rnn': RNNClassifier,
    'truncated_rnn': TruncatedRNNClassifier,
    'truncated_prosenet': TruncatedProSeNetClassifier,
    'truncated_rdt': TruncatedRDTClassifier
}

OTHER_ESTIMATORS = {
    'lr': LogisticRegression,
    'dt': DecisionTreeClassifier,
    'dummy': DummyClassifier,
    'rulefit': RuleFitClassifier,
    'riskslim': RiskSlimClassifier,
    'fasterrisk': FasterRiskClassifier,
    'frl': FRLClassifier
}

ESTIMATORS = (
    list(NET_ESTIMATORS) +
    list(RECURRENT_NET_ESTIMATORS) +
    list(OTHER_ESTIMATORS)
)
ESTIMATORS += list(map(lambda n: f'switch_{n}', ESTIMATORS))
