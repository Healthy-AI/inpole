from .models import *
from .data import *

# Each experiment maps to a dataset.
EXPERIMENTS = {
    'ra': RAData,
    'adni': ADNIData
}

NET_ESTIMATORS = {
    'sdt': SDTClassifer,
    'pronet': ProNetClassifier,
    'mlp': MLPClassifier
}

RECURRENT_NET_ESTIMATORS = {
    'rdt': RDTClassifer,
    'prosenet': ProSeNetClassifier,
    'rnn': RNNClassifier
}

OTHER_ESTIMATORS = {
    'lr': LogisticRegression,
    'dt': DecisionTreeClassifier,
    'dummy': DummyClassifier,
    'rulefit': RuleFitClassifier,
    'riskslim': RiskSlimClassifier,
    'fasterrisk': FasterRiskClassifier
}

ESTIMATORS = (
    list(NET_ESTIMATORS) +
    list(RECURRENT_NET_ESTIMATORS) +
    list(OTHER_ESTIMATORS)
)
ESTIMATORS += list(map(lambda n: f'switch_{n}', ESTIMATORS))
