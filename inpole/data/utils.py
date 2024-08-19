import re

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import skorch
from skorch.dataset import get_len
from skorch.utils import to_numpy


__all__ = [
    'get_aggregation_index',
    'check_fit_preprocessor',
    'get_feature_names',
    'drop_shifted_columns',
    'shift_variable',
    'get_shifted_features',
    'pad_pack_sequences',
    'StandardDataset',
    'SequentialDataset',
    'TruncatedHistoryDataset'
]


def get_aggregation_index(preprocessor, X, y):
    preprocessor = clone(preprocessor)
    ct = preprocessor.named_steps['column_transformer']
    ct = check_fit_preprocessor(ct, X, y)
    ct_feature_names = ct.get_feature_names_out()
    return [i for i, s in enumerate(ct_feature_names) if '_agg' in s]


def check_fit_preprocessor(preprocessor, X=None, y=None):
    if not hasattr(preprocessor, 'n_features_in_'):
        if X is None:
            raise ValueError(
                "Training data `X` must be provided if the preprocessor is" 
                "not already fitted."
            )
        if (
            isinstance(preprocessor, Pipeline)
            and preprocessor[1][0] is not None  # history aggregation
        ):
            agg_index = get_aggregation_index(preprocessor, X, y)
            preprocessor.fit(X, y, feature_selector__aggregator__agg_index=agg_index)
        else:
            preprocessor.fit(X, y)
    return preprocessor


def get_feature_names(preprocessor, X=None, y=None, trim=True):
    preprocessor = check_fit_preprocessor(preprocessor, X, y)
    feature_names = preprocessor.get_feature_names_out()
    remainders = [n for n in feature_names if n.startswith('remainder')]
    assert len(remainders) <= 1
    if trim:
        feature_names = [s.split('__')[1] for s in feature_names if not s in remainders]
    else:
        feature_names = [s for s in feature_names if not s in remainders]
    return np.array(feature_names)


def drop_shifted_columns(X):
    c_shifted = get_shifted_features(X.columns)
    return X.drop(columns=c_shifted)


def shift_variable(grouped_variable, period, fillna=None):
    shifted = grouped_variable.shift(periods=period)
    if fillna is not None:
        shifted = shifted.fillna(fillna)
    shifted.rename(f'{shifted.name}_{period}', inplace=True)
    return shifted


def get_shifted_features(features):
    p1 = re.compile(r'_[1-9]\d*$')
    p2 = re.compile(r'_[1-9]\d*_')
    return [f for f in features if p1.search(f) or p2.search(f)]


def pad_pack_sequences(batch):
    sequences, targets = zip(*batch)
    lengths = [sequence.shape[0] for sequence in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_padded_sequences = pack_padded_sequence(
        padded_sequences,
        batch_first=True,
        lengths=lengths,
        enforce_sorted=False
    )
    try:
        targets = torch.cat(targets, dim=0)
    except TypeError:
        # When using a `TruncatedHistoryDataset`, the targets are
        # one-dimensional.
        targets = torch.tensor(targets)
    except RuntimeError:
        # If there are no targets, torch.tensor(0) is used as a replacement.
        # We then need to stack instead of concatenating!
        targets = torch.stack(targets, dim=0)
    return packed_padded_sequences, targets


def _validate_dataset_inputs(X, y):
    assert isinstance(X, np.ndarray)
    if y is not None:
        assert isinstance(y, np.ndarray)


class StandardDataset(Dataset):
    def __init__(self, X, y=None):
        super(Dataset, self).__init__()
        _validate_dataset_inputs(X, y)
        self.X = X
        self.y = y
    
    def _transform(self, X, y):
        X = torch.Tensor(X)
        if y is None:
            y = torch.tensor(0)
        return X, y

    def __getitem__(self, i):
        X, y = self.X, self.y
        Xi = X[i]
        yi = y[i] if y is not None else None
        return self._transform(Xi, yi)

    def __len__(self):
        return len(self.X)


class SequentialDataset(Dataset):
    def __init__(self, X, y=None):
        """Sequential dataset for PyTorch.

        Parameters
        ----------
        X : NumPy Array of shape (n_samples, n_features)
            Input data. The last column of `X` should represent the group.
        y : NumPy Array of shape (n_samples,)
            Output targets.
        """
        super(Dataset, self).__init__()
        self._validate_parameters(X, y)
        Xg = pd.DataFrame(X)
        c_group = Xg.columns[-1]
        sequences, targets = [], []
        for _, sequence in Xg.groupby(by=c_group, sort=False):
            sequence = sequence.drop(c_group, axis=1)
            sequence = sequence.astype(np.float32)
            sequences += [sequence]
            if y is not None:
                sequence_index = list(sequence.index)
                targets += [y[sequence_index]]
        self.sequences = sequences
        self.targets = targets
    
    def _validate_parameters(self, X, y):
        assert isinstance(X, np.ndarray)
        if y is not None:
            assert isinstance(y, np.ndarray)

    def _transform(self, X, y):
        X = torch.Tensor(X.values)
        if y is None:
            y = torch.Tensor([0])
        else:
            y = torch.Tensor(y)
        y = y.type(torch.LongTensor)
        return X, y

    def __getitem__(self, i):
        X, y = self.sequences, self.targets
        Xi = X[i]
        yi = y[i] if len(y) > 0 else None
        return self._transform(Xi, yi)

    def __len__(self):
        return len(self.sequences)


class TruncatedHistoryDataset(Dataset):
    def __init__(self, X, y=None, periods=0):
        super(Dataset, self).__init__()
        _validate_dataset_inputs(X, y)
        self.X, self.y = X, y
        self.groups = X[:, -1]
        X = X[:, :-1].astype(np.float32)
        self.sequences = pd.DataFrame(X).groupby(by=self.groups)
        self.periods = periods
    
    def __getitem__(self, i):
        group = self.groups[i]
        Xi = self.sequences.get_group(group)
        Xi = Xi.loc[i-self.periods:i]
        Xi = torch.from_numpy(Xi.values)
        yi = self.y[i] if self.y is not None else torch.tensor(0)
        return Xi, yi
    
    def __len__(self):
        return len(self.X)


class ValidSplit(skorch.dataset.ValidSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, dataset, y=None, groups=None):
        if self.stratified and (y is None):
            raise ValueError(
                "Stratified cross-validation requires explicitly "
                "passing a suitable `y`."
            )

        if self.stratified and isinstance(dataset, SequentialDataset):
            raise ValueError(
                "Stratified cross-validation is not supported "
                "for a dataset of type `SequentialDataset`."
            )

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise ValueError(
                "A stratified cross-validator is required when "
                "`self.stratified=True`."
            )

        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if (
                len_dataset != len_y and
                not isinstance(dataset, SequentialDataset)
            ):
                raise ValueError(
                    "Cannot perform a CV split if `dataset` and `y` "
                    "have different lengths."
                )

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid
    