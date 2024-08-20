import copy
import math
import warnings
from functools import partial
from contextlib import contextmanager

import rulefit
import riskslim
import numpy as np
import pandas as pd
from scipy.special import expit
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from amhelpers.metrics import ece, sce

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import skorch
import skorch.callbacks as cbs
from skorch.utils import to_tensor
from skorch.dataset import unpack_data, get_len

import sklearn.linear_model as lm
import sklearn.dummy as dummy
import sklearn.calibration as calibration
import sklearn.tree as tree
import sklearn.metrics as metrics
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from FRLOptimization import (
    mine_antecedents,
    learn_FRL,
    display_rule_list
)

from .utils import plot_save_stats
from ..utils import (
    seed_torch,
    compute_squared_distances,
    get_index_per_time_step
)
from ..tree import (
    create_decision_stump,
    create_decision_tree,
    get_node_indices_per_layer
)
from .modules import (
    PrototypeNetwork,
    NNEncoder,
    RNNEncoder,
    SDT,
    RDT
)


# Include new classifers here.
__all__ = [
    'SDTClassifer',
    'RDTClassifer',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'DummyClassifier',
    'ProNetClassifier',
    'ProSeNetClassifier',
    'SwitchPropensityEstimator',
    'RuleFitClassifier',
    'RiskSlimClassifier',
    'FasterRiskClassifier',
    'MLPClassifier',
    'RNNClassifier',
    'FRLClassifier',
    'TruncatedRNNClassifier',
    'TruncatedProSeNetClassifier',
    'TruncatedRDTClassifier',
    'CalibratedClassifierCV',
]


def get_model_complexity(model):
    if isinstance(model, LogisticRegression):
        return np.count_nonzero(model.coef_)
    if isinstance(model, RiskSlimClassifier):
        return np.count_nonzero(model.coef_)
    elif isinstance(model, FRLClassifier):
        return sum(len(rule) for rule in model.rule_list if isinstance(rule, tuple))
    elif isinstance(model, RuleFitClassifier):
        return sum([len(str(r).split('&')) for r in model.rule_ensemble.rules])
        # return np.count_nonzero(model.coef_)
    elif isinstance(model, (ProNetClassifier, ProSeNetClassifier)):
        return model.module_.num_prototypes
    elif isinstance(model, (MLPClassifier, RNNClassifier)):
        return sum(p.numel() for p in model.module_.parameters() if p.requires_grad)
    elif isinstance(model, DecisionTreeClassifier):
        return model.get_n_leaves()
    elif isinstance(model, (SDTClassifer, RDTClassifer)):
        return len(model.tree_.leaf_nodes)
    else:
        raise ValueError(f"Unsupported model {type(model).__name__}.")


class EpochScoring(cbs.EpochScoring):
    # @TODO: Modify `y_pred` in SDT/RDT and remove this class.
    def on_batch_end(self, net, batch, y_pred, training, **kwargs):
        if not self.use_caching or training != self.on_train:
            return
        _, y = unpack_data(batch)
        if y is not None:
            self.y_trues_.append(y)
        if isinstance(y_pred, dict):
            self.y_preds_.append(y_pred['predictions'])
        else:
            self.y_preds_.append(y_pred)


# =============================================================================
# Base classes
# =============================================================================


class ClassifierMixin:
    accepted_metrics = {
        'auc': {'lower_is_better': False},
        'accuracy': {'lower_is_better': False},
        'ece': {'lower_is_better': True},
        'sce': {'lower_is_better': True},
        'brier': {'lower_is_better': True}
    }

    def score(self, X, y, metric='auc', **kwargs):
        if not metric in self.accepted_metrics:
            raise ValueError(
                f"Got invalid metric {metric}. "
                f"Valid metrics are {self.accepted_metrics.keys()}."
            )
        
        if y is None:
            if isinstance(X, torch.utils.data.Dataset):
                dataset = X
            else:
                dataset = self.get_dataset(X)
            y = self.collect_labels_from_dataset(dataset)
        
        if metric == 'auc':
            yp = self.predict_proba(X)
            if yp.shape[1] == 2: yp = yp[:, 1]
            if yp.ndim > 1 and not 'multi_class' in kwargs:
                kwargs['multi_class'] = 'ovr'
            try:
                return metrics.roc_auc_score(y, yp, **kwargs)
            except ValueError:
                return float('nan')
        
        if metric == 'accuracy':
            yp = self.predict(X)
            return metrics.accuracy_score(y, yp, **kwargs)

        if metric == 'brier':
            if len(self.classes_) > 2:
                # Brier score is not defined for multiclass classification.
                return float('nan')
            else:
                yp = self.predict_proba(X)[:, 1]
                return metrics.brier_score_loss(y, yp, **kwargs)

        if metric == 'ece':
            yp = self.predict_proba(X)
            if yp.shape[1] == 2: yp = yp[:, 1]
            return ece(y, yp)
        
        if metric == 'sce':
            n_classes = len(self.classes_)
            if n_classes == 2:
                # SCE is not defined for binary classification.
                return float('nan')
            else:
                y = np.eye(n_classes)[y]
                yp = self.predict_proba(X)
                return sce(y, yp)
    
    def compute_auc(self, net, X, y, **kwargs):
        return net.score(X, y, metric='auc', **kwargs)

    def compute_accuracy(self, net, X, y, **kwargs):
        return net.score(X, y, metric='accuracy', **kwargs)

    def compute_ece(self, net, X, y, **kwargs):
        return net.score(X, y, metric='ece', **kwargs)
    
    def compute_sce(self, net, X, y, **kwargs):
        return net.score(X, y, metric='sce', **kwargs)


class NeuralNetClassifier(ClassifierMixin, skorch.NeuralNetClassifier):
    prefix_ = ''

    def __init__(
        self,
        results_path,
        *,
        epoch_scoring='auc',
        monitor='auc',
        early_stopping=True,
        patience=5,
        seed=2023,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.results_path = results_path
        self.epoch_scoring = epoch_scoring
        self.monitor = monitor
        self.early_stopping = early_stopping
        self.patience = patience
        self.seed = seed

        self._validate_parameters()

        if epoch_scoring is not None:
            self._add_epoch_scoring_to_callbacks()
        
        if monitor is not None:
            self._add_checkpoint_to_callbacks()

        if early_stopping:
            self._add_early_stopping_to_callbacks()
        
        if seed is not None:
            seed_torch(seed)
    
    def _validate_parameters(self):
        epoch_scorings = [None] + list(self.accepted_metrics)
        if not self.epoch_scoring in epoch_scorings:
            raise ValueError(
                f"Expected epoch_scoring to be in {epoch_scorings}. "
                f"Got {self.epoch_scoring}."
            )
        
        monitors = set([None, 'loss', self.epoch_scoring])
        if not self.monitor in monitors:
            raise ValueError(
                f"Expected monitor to be in {monitors} "
                f"Got {self.monitor}."
            )
        
        if self.early_stopping and self.monitor is None:
            raise ValueError(
                "To enable early stopping, `monitor` must "
                "not be `None`."
            )
    
    @property
    def _keys_ignored(self):
        return 'classes_'

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', cbs.EpochTimer()),
            ('train_loss', cbs.PassthroughScoring(name='train_loss', on_train=True)),
            ('valid_loss', cbs.PassthroughScoring(name='valid_loss')),
            ('print_log', cbs.PrintLog(self._keys_ignored))
        ]

    def _add_epoch_scoring_to_callbacks(self):
        scoring = getattr(self, f'compute_{self.epoch_scoring}')
        lower_is_better = self.accepted_metrics[self.epoch_scoring]['lower_is_better']
        kwargs = {
            'scoring': scoring,
            'lower_is_better': lower_is_better,
            'use_caching': True
        }
        train = f'train_{self.epoch_scoring}'
        valid = f'valid_{self.epoch_scoring}'
        callbacks = [
            (train, EpochScoring(name=train, on_train=True, **kwargs)),
            (valid, EpochScoring(name=valid, **kwargs))
        ]
        if self.callbacks is None:
            self.callbacks = callbacks
        else:
            self.callbacks.extend(callbacks)
    
    def _add_checkpoint_to_callbacks(self):
        monitor = f'valid_{self.monitor}_best'
        callback = cbs.Checkpoint(
            monitor=monitor,
            f_params='params.pt',
            f_optimizer=None,
            f_criterion=None,
            f_history='history.json',
            f_pickle=None,
            fn_prefix='best_',
            dirname=self.results_path,
            load_best=True,
            #sink=print
        )
        callback = ('checkpoint', callback)
        if self.callbacks is None:
            self.callbacks = [callback]
        else:
            self.callbacks.append(callback)
        
    def _add_early_stopping_to_callbacks(self):
        monitor = f'valid_{self.monitor}'
        lower_is_better = True if 'loss' in monitor else \
            self.accepted_metrics[self.monitor]['lower_is_better']
        callback = cbs.EarlyStopping(
            monitor=monitor,
            lower_is_better=lower_is_better,
            patience=self.patience
        )
        callback = ('early_stopping', callback)
        self.callbacks.append(callback)
    
    def get_split_datasets(self, X, y, X_valid=None, y_valid=None, **fit_params):
        if X_valid is not None:
            dataset_train = self.get_dataset(X, y)
            dataset_valid = self.get_dataset(X_valid, y_valid)
            return dataset_train, dataset_valid
        return super().get_split_datasets(X, y, **fit_params)
    
    # Catch `fit_params` here and do not pass it on to the step function.
    def run_single_epoch(self, iterator, training, prefix, step_fn, **fit_params):
        if iterator is None:
            return

        batch_count = 0
        for batch in iterator:
            self.notify('on_batch_begin', batch=batch, training=training)
            step = step_fn(batch)
            self.history.record_batch(prefix + '_loss', step['loss'].item())
            if isinstance(batch, (tuple, list)):
                if isinstance(batch[0], PackedSequence):
                    batch_size = int(batch[0].batch_sizes[0])
                else:
                    batch_size = get_len(batch[0])
            else:
                batch_size = get_len(batch)
            self.history.record_batch(prefix + '_batch_size', batch_size)
            self.notify('on_batch_end', batch=batch, training=training, **step)
            batch_count += 1
        
        self.history.record(prefix + '_batch_count', batch_count)

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        super().on_epoch_begin(
            net, dataset_train, dataset_valid, **kwargs
        )
        if len(self.history) == 1:
            self.history.record('classes_', self.classes_.tolist())
    
    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X, y, **kwargs)

        losses = ['train_loss', 'valid_loss']
        name = self.prefix_ + 'losses'
        plot_save_stats(self.history, losses, self.results_path, name, from_batches=False)

        scores = [f'train_{self.epoch_scoring}', f'valid_{self.epoch_scoring}']
        name = self.prefix_ + self.epoch_scoring
        plot_save_stats(self.history, scores, self.results_path, name, from_batches=False)

    def collect_labels_from_dataset(self, dataset):
        y = []
        for _, yi in dataset:
            y.extend(yi) if hasattr(yi, '__iter__') \
                else y.append(yi)
        return np.array(y)

    @contextmanager
    def _current_prefix(self, prefix):
        try:
            self.prefix_ = prefix
            yield
        finally:
            self.prefix_ = ''


class SoftDecisionTreeClassifier(NeuralNetClassifier):
    def __init__(
        self,
        *,
        initial_depth=2,
        max_depth=5,
        grow_incrementally=True,
        lambda_=0.001,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.warm_start:
            warnings.warn(
                "Setting `warm_start` to `True` has no effect "
                "since `fit` is overriden."
            )
        
        if grow_incrementally:
            assert self.monitor is not None

        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.grow_incrementally = grow_incrementally
        self.lambda_ = lambda_
    
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred['predictions'][0]
        y_true = to_tensor(y_true, device=self.device)
        loss = self.criterion_(logits, y_true)
        penalty = y_pred['penalties'][0]
        loss += self.lambda_ * penalty
        return loss

    def initialize_module(self):
        kwargs = self.get_params_for('module')
        if getattr(self, 'tree_', False):
            kwargs['tree'] = self.tree_
        else:
            # Use a decision stump as a default tree
            # to avoid errors when initializing the optimizer.
            kwargs['tree'] = create_decision_stump()
        module = self.initialized_instance(self.module, kwargs)
        self.module_ = module
        return self
    
    def _set_weights(self, previous_tree, previous_params, nodes_to_optimize=[]):
        def hook(grad, mask):
            return grad * mask
    
        for name, param in self.get_all_learnable_params():
            previous_param = previous_params[name]

            new_param = []
            grad_mask = torch.zeros_like(param)

            if name.startswith('inner_nodes'):
                for node in self.tree_.sorted_inner_nodes.values():
                    if node in nodes_to_optimize:
                        index = self.tree_.inner_node_index(node)
                        grad_mask[index, :] = 1.0
                    if node.index in previous_tree.inner_nodes:
                        index = previous_tree.inner_node_index(node)
                        _new_param = previous_param[index]
                    else:
                        index = previous_tree.inner_node_index(node.parent)
                        std = math.sqrt(1 / (node.layer_index+1))
                        _new_param = previous_param[index].detach().clone()
                        noise = torch.normal(0.0, std, size=_new_param.size())
                        _new_param += noise.to(_new_param.device)
                    new_param.append(_new_param)
                new_param = torch.stack(new_param, dim=0)
            else:
                assert 'leaf_nodes' in name
                for node in self.tree_.sorted_leaf_nodes.values():
                    if node in nodes_to_optimize:
                        index = self.tree_.leaf_node_index(node)
                        grad_mask[:, index] = 1.0
                    if node.index in previous_tree.leaf_nodes:
                        index = previous_tree.leaf_node_index(node)
                        _new_param = previous_param[:, index]
                    elif hasattr(node, 'params_'):
                        _new_param = node.params_[name]
                        delattr(node, 'params_')
                    else:
                        index = self.tree_.leaf_node_index(node)
                        _new_param = param[:, index]
                    new_param.append(_new_param)
                new_param = torch.stack(new_param, dim=1)

            with torch.no_grad():
                param.copy_(new_param)
            
            if nodes_to_optimize and param.requires_grad:
                param.register_hook(partial(hook, mask=grad_mask))
    
    def _print_weights(self, s):
        print(f"Inner node weights after {s}:", self.module_.inner_nodes[0].weight)
        print(f"Leaf node weights after {s}:", self.module_.leaf_nodes[0].weight)
        print("Inner node indices:", self.tree_.inner_nodes.keys())
        print("Leaf node indices:", self.tree_.leaf_nodes.keys())
        print('\n\n\n')
    
    def fit(self, X, y, **fit_params):
        if not self.grow_incrementally:
            self.tree_ = create_decision_tree(self.max_depth)
            return super().fit(X, y, **fit_params)

        self.tree_ = create_decision_tree(depth=self.initial_depth)

        # Perform initial optimization.
        self.set_params(callbacks__checkpoint__fn_prefix=f'initial_best_')
        
        self.initialize()
        with self._current_prefix('initial_'):
            self.partial_fit(X, y, **fit_params)

        if self.verbose:
            self._print_weights("initial optimization")

        previous_tree = copy.deepcopy(self.tree_)
        previous_params = dict(self.get_all_learnable_params())

        checkpoint = dict(self.callbacks_).get('checkpoint')
        assert checkpoint.load_best
        scoring = checkpoint.monitor.replace('_best', '')
        lower_is_better = True if 'loss' in self.monitor else \
            self.accepted_metrics[self.monitor]['lower_is_better']
        best_valid_score = self.history[-1][scoring]

        iteration = 1

        while len(self.tree_.suboptimal_leaves) > 0:
            self.set_params(
                callbacks__checkpoint__fn_prefix=f'{iteration:02d}_best_'
            )

            suboptimal_leaf = self.tree_.suboptimal_leaves.pop()
            if suboptimal_leaf.layer_index == self.max_depth:
                break

            # Store the parameters of the suboptimal leaf in case
            # we will retain it as optimal.
            params = {}
            index = previous_tree.leaf_node_index(suboptimal_leaf)
            for name, param in previous_params.items():
                if 'leaf_nodes' in name:
                    params[name] = param[:, index].detach().clone()
            suboptimal_leaf.params_ = params

            # Split suboptimal leaf into inner node with two leaves.
            self.tree_.split_node(index=suboptimal_leaf.index)

            nodes_to_optimize = [suboptimal_leaf]
            nodes_to_optimize += suboptimal_leaf.children

            self.initialize()
            self._set_weights(
                previous_tree, previous_params, nodes_to_optimize=nodes_to_optimize
            )
            with self._current_prefix(f'{iteration:02d}_'):
                self.partial_fit(X, y, **fit_params)

            if self.verbose:
                self._print_weights(f"iteration {iteration}")
            
            previous_tree = copy.deepcopy(self.tree_)
            previous_params = dict(self.get_all_learnable_params())

            valid_score = self.history[-1][scoring]
            improved = valid_score < best_valid_score if lower_is_better else \
                valid_score > best_valid_score
            if improved:
                best_valid_score = valid_score
            else:
                self.tree_.unsplit_node(index=suboptimal_leaf.index)

            iteration += 1
        
        # Perform global optimization.
        self.set_params(callbacks__checkpoint__fn_prefix=f'final_best_')

        self.initialize()
        self._set_weights(previous_tree, previous_params)
        with self._current_prefix('final_'):
            self.partial_fit(X, y, **fit_params)
        
        if self.verbose:
            self._print_weights("global optimization")

        return self

    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            y_pred = self.infer(Xi, predict_only=True)
            return (
                y_pred['predictions'][0],  # logits
                y_pred['probas'][1],  # all path probabilities
                *y_pred['other']  # thresholds
            )
    
    def align_axes(self):
        self.module_._align_axes()
    
    def prune_tree(self, all_path_probas, pruning_threshold=0.05):
        average_path_probas = all_path_probas.mean(dim=0)
        eliminate_node = (average_path_probas < pruning_threshold).tolist()
        self.module_._perform_node_pruning(eliminate_node)
    
    def draw_tree(
        self,
        features,
        labels,
        all_path_probas=None,
        thresholds=None,
        X=None,
        index=None,
        max_width=5
    ):
        if all_path_probas is None:
            if X is None:
                raise ValueError(
                    "Input argument `X` must not be `None` when "
                    "`all_path_probas=None`."
                )
            # Avoid shuffling the data by setting `training=False`.
            _, all_path_probas, thresholds = self.forward(X, training=False)
        
        tree_depth = self.module_.tree.depth
        node_indices_per_layer = get_node_indices_per_layer(tree_depth+1)

        edge_attrs = {}
        for i_layer, node_indices in node_indices_per_layer.items():
            if i_layer == 0:
                continue
            path_probas = all_path_probas[:, node_indices]
            mean_path_probas = path_probas[index].mean(axis=0)
            for i, p in zip(node_indices, mean_path_probas):
                if not p.isnan():
                    p = p.item()
                    width = str(np.interp(p, [0, 1], [1, max_width]))
                    style = '' if p > 0 else 'dashed'
                    label = f'{100 * p:.1f}%'
                    edge_attr = {
                        'penwidth': width,
                        'style': style,
                        'label': label
                    }
                    edge_attrs[i] = edge_attr
        
        if thresholds is not None:
            thresholds = thresholds[index]
        
        return self.module_._draw_tree(features, labels, thresholds, edge_attrs=edge_attrs)
    
    def save_tree(self, features, labels, suffix=''):
        graph = self.module_._draw_tree(features, labels)
        graph.render('tree%s' % suffix, self.results_path, view=False, format='png')


class PrototypeClassifier(NeuralNetClassifier):
    loss_terms = [
        'ce_loss',
        'div_loss',
        'cl_loss',
        'ev_loss'
    ]

    def __init__(
        self,
        *,
        d_min=1,
        lambda_div=1e-3,
        lambda_ev=1e-3,
        lambda_cl=1e-3,
        projection_interval=5,
        **kwargs
    ):
        super(PrototypeClassifier, self).__init__(**kwargs)

        self.d_min = d_min
        self.lambda_div = lambda_div
        self.lambda_ev = lambda_ev
        self.lambda_cl = lambda_cl

        self.projection_interval = projection_interval
    
    @property
    def _keys_ignored(self):
        return [
            'classes_',
            'prototype_indices',
            'prototypes'
        ]

    def _project_prototypes(self, X):
        # Avoid shuffling the data by setting `training=False`.
        _, similarities, encodings = self.forward(X, training=False)
        _, max_indices = torch.max(similarities, dim=0)
        projections = encodings[max_indices]
        return projections, max_indices

    def _project_and_record_prototypes(self, dataset_train):
        assert hasattr(self.module_, 'prototype_layer')
        projections, max_indices = self._project_prototypes(dataset_train)
        self.history.record('prototype_indices', max_indices.cpu().tolist())
        self.module_.set_prototypes(projections)
    
    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss_terms = self._get_loss_terms(y_pred, yi, Xi, training=False)
        tot_loss = sum(loss_terms.values())
        loss_terms.update(
            {'loss': tot_loss, 'y_pred': y_pred}
        )
        return loss_terms

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss_terms = self._get_loss_terms(y_pred, yi, Xi, training=False)
        tot_loss = sum(loss_terms.values())
        tot_loss.backward()
        loss_terms.update(
            {'loss': tot_loss, 'y_pred': y_pred}
        )
        return loss_terms
    
    def on_batch_end(self, net, batch=None, training=False, **kwargs):
        prefix = 'train_' if training else 'valid_'
        for loss_term in self.loss_terms:
            self.history.record_batch(prefix + loss_term, kwargs[loss_term].item())

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        if (
            self.projection_interval > 0 and
            self.history[-1]['epoch'] % self.projection_interval == 0
        ):
            self._project_and_record_prototypes(dataset_train)

    def _get_loss_terms(self, y_pred, y_true, X, training):
        loss_terms = {}
        
        outputs, _, encodings = y_pred
        
        # Compute cross-entropy loss.
        ce_loss = super().get_loss(outputs, y_true, X, training)
        loss_terms['ce_loss'] = ce_loss

        if hasattr(self.module_, 'prototype_layer'):
            prototypes = self.module_.prototype_layer.prototypes
            if not training:
                prototypes = prototypes.detach()
                encodings = encodings.detach()
            pdistances = nn.functional.pdist(prototypes).unsqueeze(0)
            sq_distances = compute_squared_distances(encodings, prototypes)

            # Compute diversity regularization.
            zeros = torch.zeros_like(pdistances)
            temp = torch.cat([zeros, self.d_min - pdistances], dim=0)
            div_reg = torch.square(torch.max(temp, dim=0)[0]).sum()
            loss_terms['div_loss'] = self.lambda_div * div_reg

            # Compute clustering regularization.
            cl_reg = sq_distances.min(dim=1)[0].sum()
            loss_terms['cl_loss'] = self.lambda_cl * cl_reg

            # Compute evidence regularization.
            ev_reg = sq_distances.min(dim=0)[0].sum()
            loss_terms['ev_loss'] = self.lambda_ev * ev_reg
        
        return loss_terms
    
    def partial_fit(self, X, y=None, classes=None, **fit_params):
        super().partial_fit(X, y, classes, **fit_params)

        # Check if we need to project the prototypes again.
        #
        # Note that we cannot perform this step in `self.on_train_end`
        # because it is called before any callback that possibly
        # loads the best model.
        if (
            self.projection_interval > 0 and
            not 'prototypes' in self.history[-1]
        ):
            # If `self.train_split = None`, we can directly pass `X`
            # to `self._project_and_record_prototypes`, but it does
            # not hurt to extract the training dataset.
            dataset_train, _ = self.get_split_datasets(X, y, **fit_params)
            self._project_and_record_prototypes(dataset_train)

        return self

    def on_train_end(self, net, X=None, y=None, **kwargs):
        super(PrototypeClassifier, self).on_train_end(net, X, y, **kwargs)
        for mode in ['train', 'valid']:
            loss_terms = [f'{mode}_{loss_term}' for loss_term in self.loss_terms]
            name = self.prefix_ + mode + '_penalties'
            plot_save_stats(self.history, loss_terms, self.results_path, name)


# =============================================================================
# Public estimators
# =============================================================================


class SDTClassifer(SoftDecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(module=SDT, **kwargs)


class RDTClassifer(SoftDecisionTreeClassifier):
    def __init__(self, *, delta1=0.001, delta2=0.001, **kwargs):
        super().__init__(module=RDT, **kwargs)

        self.delta1 = delta1
        self.delta2 = delta2

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        
        path_probas = y_pred['probas'][0]

        batch_size = y_true.shape[0]
        input = self.module_.leaf_nodes[0].weight.expand(batch_size, -1, -1)  # Shape (batch_size, n_classes, n_leaf_nodes)
        
        num_leaf_nodes = path_probas.shape[-1]
        target = y_true.unsqueeze(dim=1).expand(batch_size, num_leaf_nodes)

        ce = torch.nn.functional.cross_entropy(input, target, reduction='none')

        loss = torch.sum(path_probas * ce, dim=-1).mean()
        loss += self.delta1 * y_pred['penalties'][0]
        loss += self.delta2 *  y_pred['penalties'][1]
        loss += self.lambda_ *  y_pred['penalties'][2]
        return loss

    def on_batch_end(self, net, batch=None, training=False, **kwargs):
        prefix = 'train' if training else 'valid'

        y_pred  = kwargs['y_pred']
        
        logits = y_pred['predictions'][0]
        probas = torch.nn.functional.softmax(logits, dim=1)
        probas_np = probas.detach().cpu().numpy()
        
        for i, c in enumerate(self.classes_):
            self.history.record_batch(
                f'{prefix}_proba_{c}', probas_np[:, i].mean().item()
            )

        penalties = ['evolution_penalty', 'behavior_penalty', 'splitting_penalty']
        for i, penalty in enumerate(penalties):
            self.history.record_batch(
                f'{prefix}_{penalty}', y_pred['penalties'][i].item()
            )

    def on_train_end(self, net, X=None, y=None, **kwargs):
        super(RDTClassifer, self).on_train_end(net, X, y, **kwargs)
        for mode in ['train', 'valid']:
            probas = [f'{mode}_proba_{c}' for c in self.classes_]
            name = self.prefix_ + mode + '_probas'
            plot_save_stats(self.history, probas, self.results_path, name)

            penalties = ['evolution_penalty', 'behavior_penalty', 'splitting_penalty']
            penalties = [f'{mode}_{penalty}' for penalty in penalties]
            name = self.prefix_ + mode + '_penalties'
            plot_save_stats(self.history, penalties, self.results_path, name)
        
    def draw_all_trees(self, features, labels, X, groups):    
        # Avoid shuffling the data by setting `training=False`.
        _, all_path_probas, thresholds = self.forward(X)
        
        graphs = []
        for index_per_time_step in get_index_per_time_step(groups):
            graphs += [
                self.draw_tree(
                    features,
                    labels,
                    all_path_probas=all_path_probas,
                    thresholds=thresholds,
                    index=index_per_time_step
                )
            ]
        
        return graphs


class LogisticRegression(ClassifierMixin, lm.LogisticRegression):
    def __init__(
        self,
        penalty='l2',
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=100,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )


class DecisionTreeClassifier(ClassifierMixin, tree.DecisionTreeClassifier):
    def __init__(
        self,
        *,
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        num_ccp_alphas=10
    ):
        self.num_ccp_alphas = num_ccp_alphas
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha
        )

    def fit(self, X, y, **fit_params):
        X_valid = fit_params.pop('X_valid', None)
        y_valid = fit_params.pop('y_valid', None)
        if (
            X_valid is not None
            and y_valid is not None
            and self.num_ccp_alphas is not None
        ):
            path = self.cost_complexity_pruning_path(X, y)
            indices = np.linspace(0, len(path.ccp_alphas) - 2,
                                  self.num_ccp_alphas, dtype=int)
            best_ccp_alpha, best_score = None, -np.inf
            for ccp_alpha in path.ccp_alphas[indices]:
                self.set_params(ccp_alpha=ccp_alpha)
                super().fit(X, y, **fit_params)
                score = self.score(X_valid, y_valid)
                print(f'ccp_alpha: {ccp_alpha:.5f}', f'score: {score:.2f}')
                if score > best_score:
                    best_score = score
                    best_ccp_alpha = ccp_alpha
            self.set_params(ccp_alpha=best_ccp_alpha)
            return super().fit(X, y, **fit_params)
        else:
            return super().fit(X, y, **fit_params)


class DummyClassifier(ClassifierMixin, dummy.DummyClassifier):
    def __init__(
        self, *, strategy='prior', random_state=None, constant=None
    ):
        super().__init__(
            strategy=strategy,
            random_state=random_state,
            constant=constant
        )


class ProNetClassifier(PrototypeClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            module=PrototypeNetwork,
            module__encoder=NNEncoder,
            **kwargs
        )


class ProSeNetClassifier(PrototypeClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            module=PrototypeNetwork,
            module__encoder=RNNEncoder,
            **kwargs
        )


class MLPClassifier(NeuralNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            module=PrototypeNetwork,
            module__encoder=NNEncoder,
            module__num_prototypes=-1,
            **kwargs
        )
    
    def infer(self, x, **fit_params):
        return super().infer(x, **fit_params)[0]


class RNNClassifier(NeuralNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            module=PrototypeNetwork,
            module__encoder=RNNEncoder,
            module__num_prototypes=-1,
            **kwargs
        )
    
    def infer(self, x, **fit_params):
        return super().infer(x, **fit_params)[0]


class TruncatedRNNClassifier(RNNClassifier):
    def infer(self, x, **fit_params):
        y_pred = super().infer(x, **fit_params)
        _, lengths = pad_packed_sequence(x, batch_first=True)
        index = np.cumsum(lengths) - 1
        return y_pred[index]


class TruncatedProSeNetClassifier(ProSeNetClassifier):
    def infer(self, x, **fit_params):
        outputs, similarities, encodings = super().infer(x, **fit_params)
        _, lengths = pad_packed_sequence(x, batch_first=True)
        index = np.cumsum(lengths) - 1
        return outputs[index], similarities[index], encodings[index]


class TruncatedRDTClassifier(RDTClassifer):
    def infer(self, x, **fit_params):
        y_pred = super().infer(x, **fit_params)
        _, lengths = pad_packed_sequence(x, batch_first=True)
        index = np.cumsum(lengths) - 1
        y_pred_out = {}
        for key, values in y_pred.items():
            temp = []
            for value in values:
                if value.ndim > 0:
                   temp.append(value[index])
                else:
                    temp.append(value)
            y_pred_out[key] = tuple(temp)
        return y_pred_out


class SwitchPropensityEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator_s=None, estimator_t=None):        
        if estimator_s is None:
            estimator_s = tree.DecisionTreeClassifier()
        if estimator_t is None:
            estimator_t = tree.DecisionTreeClassifier()
        
        self.estimator_s = estimator_s
        self.estimator_t = estimator_t
    
    @property
    def estimators(self):
        """Return the (unfitted) estimators."""
        return [self.estimator_s, self.estimator_t]
    
    def _split_data(self, X, y, prev_therapy_index):
        y_prev = np.argmax(X[:, prev_therapy_index], axis=1)
        y_switch = 1 * (y_prev != y)
        switch = y_switch > 0
        return X, y_switch, X[switch], y[switch]
    
    def fit(self, X, y, prev_therapy_index, X_valid=None, y_valid=None):
        """Fit model.
        
        We assume that the labels `y` are encoded into values between 0 and 
        n_classes-1. We also assume that `prev_therapy_index` contains 
        n_classes elements.
        """

        self._check_fit_args(X, y, prev_therapy_index)
        
        self.prev_therapy_index_ = prev_therapy_index

        Xs, ys, Xt, yt = self._split_data(X, y, prev_therapy_index)
        
        fit_params_s, fit_params_t = {}, {}
        if X_valid is not None and y_valid is not None:
            Xs_valid, ys_valid, Xt_valid, yt_valid = self._split_data(
                X_valid, y_valid, prev_therapy_index
            )
            fit_params_s['X_valid'] = Xs_valid
            fit_params_s['y_valid'] = ys_valid
            fit_params_t['X_valid'] = Xt_valid
            fit_params_t['y_valid'] = yt_valid
        
        self.estimator_s_ = self.estimator_s.fit(Xs, ys, **fit_params_s)
        self.estimator_t_ = self.estimator_t.fit(Xt, yt, **fit_params_t)
        
        self.classes_ = self.estimator_t_.classes_
        
        return self

    def _check_fit_args(self, X, y, prev_therapy_index):
        # Check assumptions.
        n_classes = len(prev_therapy_index)
        assert np.array_equal(np.unique(y), np.arange(n_classes))

        # Check inputs `X`.
        assert isinstance(X, np.ndarray)

        # Check labels `y`.
        assert isinstance(y, np.ndarray)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("`y` must be numeric.")
        
        # Check indexes.
        assert isinstance(prev_therapy_index, np.ndarray)
        if not np.issubdtype(prev_therapy_index.dtype, np.integer):
            raise ValueError("`prev_therapy_index` must be an integer array.")
    
    def predict_proba(self, X):
        check_is_fitted(self)

        # Make predictions.
        y_sp = self.estimator_s_.predict_proba(X)[:, 1].reshape(-1, 1)
        y_tp = self.estimator_t_.predict_proba(X)

        # Extract the previous treatment.
        y_prev = X[:, self.prev_therapy_index_].astype(np.float32)
        
        # Remove probability of the previous treatment and renormalize.
        #
        # @TODO: Should we perhaps use a prior instead of equal probabilities below?
        y_tp = (1-y_prev) * y_tp
        mask = y_tp.sum(axis=1) == 0
        y_tp[mask] = 1 - y_prev[mask]  # Assign equal probability to all treatments except the previous one
        y_tp = y_tp / y_tp.sum(axis=1, keepdims=True)
        
        # Mix in probability of staying.
        y_p = (1-y_sp)*y_prev + y_sp*y_tp
        
        y_p_sum = y_p.sum(axis=1)
        assert np.allclose(y_p_sum, np.ones_like(y_p_sum))
        
        return y_p
    
    def predict(self, X):
        yp = self.predict_proba(X)
        return np.argmax(yp, axis=1)


class RuleFitClassifier(ClassifierMixin, rulefit.RuleFit):
    def __init__(
        self,
        tree_size=4,
        sample_fract='default',
        max_rules=50,
        memory_par=0.01,
        tree_generator=None,
        rfmode='classify',
        lin_trim_quantile=0.025,
        lin_standardise=True,
        exp_rand_tree_size=True,
        model_type='rl',
        Cs=None,
        cv=3,
        tol=0.0001,
        max_iter=100,
        n_jobs=None,
        random_state=None
    ):
        super().__init__(
            tree_size=tree_size,
            sample_fract=sample_fract,
            max_rules=max_rules,
            memory_par=memory_par,
            tree_generator=tree_generator,
            rfmode=rfmode,
            lin_trim_quantile=lin_trim_quantile,
            lin_standardise=lin_standardise,
            exp_rand_tree_size=exp_rand_tree_size,
            model_type=model_type,
            Cs=Cs,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def fit(self, X, y, feature_names=None, **kwargs):
        super().fit(X, y, feature_names, **kwargs)
        self.classes_ = np.unique(y)
        return self

    def get_rules(self, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True, returns only the rules with an estimated coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """
        n_features= len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        # Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.coef_[i]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.coef_[i]
            if subregion is None:
                importance = abs(coef)*self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in self.winsorizer.trim(subregion) ] - self.mean[i]))/len(subregion)
            output_rules += [(self.feature_names[i], 'linear',coef, 1, importance)]
        # Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.coef_[i + n_features]
            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules


class RiskSlimClassifier(ClassifierMixin):
    def __init__(
        self,
        max_coefficient=5,
        max_L0_value=5,
        max_offset=50,
        c0_value=1e-6,
        w_pos=1.00,
        random_state=None
    ):
        self.max_coefficient = max_coefficient
        self.max_L0_value = max_L0_value
        self.max_offset = max_offset
        
        self.c0_value = c0_value
        self.w_pos = w_pos
        self.random_state = random_state

        self.settings = {
            'c0_value': c0_value,
            'w_pos': w_pos,
            # =================================================================
            # LCPA settings.
            # =================================================================
            'max_runtime': 30.0,
            'max_tolerance': np.finfo('float').eps,
            'display_cplex_progress': True,
            'loss_computation': 'fast',
            # =================================================================
            # LCPA improvements.
            # =================================================================
            'round_flag': True,
            'polish_flag': True,
            'chained_updates_flag': True,
            'add_cuts_at_heuristic_solutions': True,
            # =================================================================
            # Initialization.
            # =================================================================
            'initialization_flag': True,
            'init_max_runtime': 120.0,
            'init_max_coefficient_gap': 0.49,
            # =================================================================
            # CPLEX solver.
            # =================================================================
            'cplex_randomseed': random_state,
            'cplex_mipemphasis': 0
        }
    
    def fit(self, X, y, feature_names, outcome_name):
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name=outcome_name)
        Xy = pd.concat([y, X], axis=1)
        data = riskslim.utils.prepare_data(Xy)
        
        cofficient_set = riskslim.CoefficientSet(
            variable_names=data['variable_names'],
            lb=-self.max_coefficient,
            ub=self.max_coefficient,
            sign=0
        )
        cofficient_set.update_intercept_bounds(
            X=data['X'],
            y=data['Y'],
            max_offset=self.max_offset
        )
        
        constraints = {
            'L0_min': 0,
            'L0_max': self.max_L0_value,
            'coef_set': cofficient_set
        }
        model_info, _mip_info, _lcpa_info = riskslim.run_lattice_cpa(
            data, constraints, self.settings
        )
        
        self.intercept_ = model_info['solution'][0]
        self.coef_ = model_info['solution'][1:]
        self.classes_ = np.unique(y)
        try:
            risk_table = riskslim.utils.print_model(model_info['solution'], data)
            self.risk_list = risk_table.get_string()
        except ValueError:
            self.risk_list = None
    
    def decision_function(self, X):
        return self.intercept_ + np.dot(X, self.coef_)

    def predict_proba(self, X):
        probas = self.decision_function(X)
        expit(probas, out=probas)
        return np.vstack([1 - probas, probas]).T
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


class FasterRiskClassifier(ClassifierMixin):
    def __init__(
        self,
        sparsity=5,
        lb=-5,
        ub=5,
        parent_size=10,
        random_state=None  # For compatibility, currently unused
    ):
        self.sparsity = sparsity
        self.lb = lb
        self.ub = ub
        self.parent_size = parent_size
    
    def _check_y(self, y):
        if set(y) == {0, 1}:
            y[y==0] = -1
        # We do not perform any other checks here because `y` will be checked 
        # in `RiskScoreOptimizer`.
        return y
    
    def fit(self, X, y, feature_names):
        y = self._check_y(y)
        m = RiskScoreOptimizer(
            X=X,
            y=y,
            k=self.sparsity,
            lb=self.lb,
            ub=self.ub,
            parent_size=self.parent_size
        )
        m.optimize()
        
        multiplier, intercept, coefficients = m.get_models(model_index=0)
        self.clf_ = RiskScoreClassifier(
            multiplier=multiplier,
            intercept=intercept,
            coefficients=coefficients,
            featureNames=feature_names
        )
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        probas = self.clf_.predict_prob(X)
        return np.vstack([1 - probas, probas]).T
    
    def predict(self, X):
        return self.clf_.predict(X)


class FRLClassifier(ClassifierMixin):
    def __init__(
        self,
        minsupport=10,
        max_predicates_per_ant=2,
        w=7,
        C=0.000001,
        prob_terminate=0.01,
        T=3000,
        lambda_=0.8,
        random_state=None  # For compatibility, currently unused
    ):
        self.minsupport = minsupport
        self.max_predicates_per_ant = max_predicates_per_ant
        self.w = w
        self.C = C
        self.prob_terminate = prob_terminate
        self.T = T
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rule_list = []
        self.prob_list = []

    def _encode_categorical(self, X, feature_names):
        assert X.shape[1] == len(feature_names)
        encoded_features = []
        for i, feature_name in enumerate(feature_names):
            e = [f'{feature_name}={x}' for x in X[:, i]]
            encoded_features.append(e)
        return np.column_stack(encoded_features)

    def _encode_binned(self, X, feature_names, bin_edges):
        assert X.shape[1] == len(bin_edges)
        encoded_features = []
        iterable  = zip(feature_names, bin_edges)
        for i, (feature_name, _bin_edges) in enumerate(iterable):
            indices = np.digitize(X[:, i], _bin_edges)
            starts = np.take(_bin_edges, indices - 1)
            ends = np.take(_bin_edges, indices)
            e = [f'{feature_name}=[{a},{b})' for a, b in zip(starts, ends)]
            encoded_features.append(e)
        return np.column_stack(encoded_features)
    
    def _encode_features(self, preprocessor, Xt):
        from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

        assert list(preprocessor.named_steps) == \
            ['column_transformer', 'feature_selector']
        
        if preprocessor.named_steps['feature_selector'] is not None:
            fs = preprocessor.named_steps['feature_selector']
            assert fs.n_features_in_ == len(fs.get_feature_names_out())

        assert np.array_equal(Xt, Xt.astype(bool))

        column_transformer = preprocessor.named_steps['column_transformer']
        transformers = column_transformer.transformers_
        output_indices = column_transformer.output_indices_

        transformers = [t for t in transformers if t[0] != 'remainder']

        assert all(
            list(pipeline.named_steps) == ['imputer', 'encoder']
            for _, pipeline, _ in transformers
        )

        X_encoded = []
        
        for transformer_name, pipeline, columns in transformers:
            _output_indices = output_indices[transformer_name]
            _Xt = Xt[:, _output_indices]
            encoder = pipeline.steps[-1][1]
            if encoder is None:
                encoded = self._encode_categorical(_Xt, columns)
                continue
            _X = encoder.inverse_transform(_Xt)
            if isinstance(encoder, KBinsDiscretizer):
                encoded = self._encode_binned(_X, columns, encoder.bin_edges_)
            elif isinstance(encoder, OneHotEncoder):
                encoded = self._encode_categorical(_X, columns)
            else:
                raise NotImplementedError
            X_encoded.append(encoded)
        
        return np.column_stack(X_encoded).tolist()
    
    def fit(self, X, y, preprocessor):
        print("Encoding features.")
        X = self._encode_features(preprocessor, X)

        print("Mining rules using FP-growth.")
        X_pos, X_neg, _, _, antecedent_set = \
            mine_antecedents(X, y, self.minsupport, self.max_predicates_per_ant)

        print("Running FRL algorithm.")        
        n = len(X)
        FRL_rule, FRL_prob, FRL_pos_cnt, FRL_neg_cnt, FRL_obj_per_rule, \
        FRL_Ld, FRL_Ld_over_iters, FRL_Ld_best_over_iters = learn_FRL(
            X_pos, X_neg, n, self.w, self.C, self.prob_terminate, self.T, self.lambda_
        )

        display_rule_list(FRL_rule, FRL_prob, antecedent_set, FRL_pos_cnt, 
                          FRL_neg_cnt, FRL_obj_per_rule, FRL_Ld)
        self.classes_ = np.unique(y)
        
        # Extract rules from FRL
        rule_list = []
        prob_list = []
        for i, rule_index in enumerate(FRL_rule):
            rule = antecedent_set[rule_index]
            rule_list.append(rule)
            prob = FRL_prob[i]
            prob_list.append(prob)

        self.rule_list = rule_list
        self.prob_list = prob_list
        self.preprocessor = preprocessor

    def predict_proba(self, X):
        X = self._encode_features(self.preprocessor, X)
        y_pred = np.zeros((len(X), 2))

        for i, row in enumerate(X):
            rule_index = self.check_rules(row, self.rule_list)
            if rule_index is not None:
                prob_1 = self.prob_list[rule_index]
            else:
                prob_1 = self.prob_list[-1]
            prob_0 = 1-prob_1
            y_pred[i] = [prob_0, prob_1]
        return y_pred

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return [1 if prob[1] > 0.5 else 0 for prob in y_prob]

    def check_rules(self, row, rules):
        row_set = set(row)
        for index, rule in enumerate(rules):
            if isinstance(rule, tuple):
                if all(item in row_set for item in rule):
                    return index
            elif rule in row_set:
                return index
        return None


class CalibratedClassifierCV(ClassifierMixin, calibration.CalibratedClassifierCV):
    def __init__(
        self,
        estimator=None,
        *,
        method='sigmoid',
        cv=None,
        n_jobs=None,
        ensemble=True,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            ensemble=ensemble
        )
