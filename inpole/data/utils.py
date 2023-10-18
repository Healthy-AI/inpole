import torch
import skorch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from skorch.dataset import get_len
from skorch.utils import to_numpy
from amhelpers.amhelpers import get_class_from_str


def get_data_splits_from_config(config):
    data_handler = get_class_from_str(config['data']['handler'])
    return data_handler.get_splits(
        config['data']['path'],
        config['data']['valid_size'],
        config['data']['test_size'],
        config['data']['seed']
    )


def get_classes_from_config(config):
    data_handler = get_class_from_str(config['data']['handler'])
    le = data_handler.get_splits(
        config['data']['path'],
        config['data']['valid_size'],
        config['data']['test_size'],
        config['data']['seed'],
        return_label_encoder=True
    )[-1]
    return le.classes_


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
    targets = torch.cat(targets, dim=0)
    return packed_padded_sequences, targets


class StandardDataset(Dataset):
    def __init__(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
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
        for _, sequence in Xg.groupby(by=c_group):
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
    