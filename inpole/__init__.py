from .models import *
from .data import *

# Each experiment maps to a dataset.
EXPERIMENTS = {
    'ra': RAData
}

NET_ESTIMATORS = {
    'sdt': SDTClassifer,
    'pronet': PrototypeClassifier
}

RECURRENT_NET_ESTIMATORS = {
    'rdt': RDTClassifer,
    'prosenet': PrototypeClassifier
}

OTHER_ESTIMATORS = {
    'lr': LogisticRegression,
    'dt': DecisionTreeClassifier,
    'dummy': DummyClassifier
}

NET_MODULES = {
    'sdt': SDT,
    'rdt': RDT,
    'pronet': PrototypeNetwork,
    'prosenet': PrototypeNetwork
}

ESTIMATORS = (
    list(NET_ESTIMATORS) +
    list(RECURRENT_NET_ESTIMATORS) +
    list(OTHER_ESTIMATORS)
)
ESTIMATORS += list(map(lambda n: f'switch_{n}', ESTIMATORS))
