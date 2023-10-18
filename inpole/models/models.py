import copy
import math
import warnings
import torch
import numpy as np
import torch.nn as nn
import sklearn.linear_model as lm
import sklearn.dummy as dummy
import sklearn.tree as tree
import sklearn.metrics as metrics
import skorch.callbacks as cbs
from torch.nn.utils.rnn import PackedSequence, unpack_sequence
from skorch import NeuralNetClassifier
from skorch.utils import to_tensor
from skorch.dataset import unpack_data, get_len
from .utils import plot_save_stats
from ..utils import (
    seed_torch,
    create_decision_stump,
    create_decision_tree,
    compute_squared_distances
)
from functools import partial
from contextlib import contextmanager
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


__ALL__ = [
    'SDTClassifer',
    'RDTClassifer',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'DummyClassifier',
    'PrototypeClassifier',
    'SwitchPropensityEstimator'
]


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


class ClassifierMixin:
    accepted_metrics = ['auc', 'accuracy']

    def score(self, X, y, metric='auc', **kwargs):
        if not metric in self.accepted_metrics:
            raise ValueError(
                f"Got invalid metric {metric}. "
                f"Valid metrics are {self.accepted_metrics}."
            )
        
        if y is None:
            if isinstance(X, torch.utils.data.Dataset):
                dataset = X
            else:
                dataset = self.get_dataset(X)
            y = self.collect_labels_from_dataset(dataset)
        
        if metric == 'auc':
            yp = self.predict_proba(X)
            if yp.shape[1] == 2:
                yp = yp[:, 1]
            if yp.ndim > 1 and not kwargs.pop('multi_class', False):
                kwargs['multi_class'] = 'ovr'
            try:
                return metrics.roc_auc_score(y, yp, **kwargs)
            except ValueError:
                return float('nan')
        
        if metric == 'accuracy':
            yp = self.predict(X)
            if kwargs.pop('average', False):
                return float('nan')
            else:
                return metrics.accuracy_score(y, yp, **kwargs)
    
    def compute_auc(self, net, X, y, **kwargs):
        return net.score(X, y, metric='auc', **kwargs)

    def compute_accuracy(self, net, X, y, **kwargs):
        return net.score(X, y, metric='accuracy', **kwargs)


class BaseClassifier(ClassifierMixin, NeuralNetClassifier):
    file_name_prefix_ = ''

    def __init__(
        self,
        results_path,
        *args,
        epoch_scoring='auc',
        monitor='loss',
        seed=2023,
        **kwargs
    ):
        super(BaseClassifier, self).__init__(*args, **kwargs)

        self.results_path = results_path
        self.epoch_scoring = epoch_scoring
        self.monitor = monitor
        self.seed = seed

        self._validate_parameters()

        self._add_epoch_scoring_to_callbacks()
        
        if monitor is not None:
            self._add_checkpoint_to_callbacks()
        
        seed_torch(seed)
    
    def _validate_parameters(self):
        if not self.epoch_scoring in self.accepted_metrics:
            raise ValueError(
                f"Expected epoch_scoring to be in {self.accepted_metrics}. "
                f"Got {self.epoch_scoring}."
            )
        
        if not self.monitor in [None, 'loss', self.epoch_scoring]:
            raise ValueError(
                "Expected monitor to be either `None`, 'loss' or "
                f"{self.epoch_scoring}. Got {self.monitor}."
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
        kwargs = {
            'scoring': scoring,
            'lower_is_better': False,
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
    
    def get_split_datasets(self, X, y=None, X_valid=None, y_valid=None, **fit_params):
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
        super(BaseClassifier, self).on_epoch_begin(
            net, dataset_train, dataset_valid, **kwargs
        )
        if len(self.history) == 1:
            self.history.record('classes_', self.classes_.tolist())
    
    def on_train_end(self, net, X=None, y=None, **kwargs):
        super(BaseClassifier, self).on_train_end(net, X, y, **kwargs)

        losses = ['train_loss', 'valid_loss']
        name = self.file_name_prefix_ + 'losses'
        plot_save_stats(self.history, losses, self.results_path, name, from_batches=False)

        scores = [f'train_{self.epoch_scoring}', f'valid_{self.epoch_scoring}']
        name = self.file_name_prefix_ + self.epoch_scoring
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
            self.file_name_prefix_ = prefix
            yield
        finally:
            self.file_name_prefix_ = ''


class SDTClassifer(BaseClassifier):
    def __init__(
        self,
        *args,
        initial_depth=2,
        max_depth=5,
        grow_incrementally=True,
        lambda_=0.001,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

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
                        _new_param += torch.normal(0.0, std, size=_new_param.size())
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
    
    def fit(self, X, y=None, **fit_params):
        if not self.grow_incrementally:
            self.tree_ = create_decision_tree(self.max_depth)
            return super().fit(X, y, **fit_params)

        self.tree_ = create_decision_tree(depth=self.initial_depth)

        # Perform initial optimization.
        self.set_params(
            callbacks__checkpoint__fn_prefix=f'initial_best_'
        )
        
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
        best_valid_score = self.history[-1][scoring]

        iteration = 1

        while len(self.tree_.suboptimal_leaves) > 0:
            self.set_params(
                callbacks__checkpoint__fn_prefix=f'{iteration:02d}_best_'
            )

            suboptimal_leaf = self.tree_.suboptimal_leaves.pop()
            if suboptimal_leaf.layer_index == self.max_depth:
                #assert all([node.layer_index >= self.max_depth for node in self.tree_.suboptimal_leaves])
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
            self._set_weights(previous_tree, previous_params, nodes_to_optimize=nodes_to_optimize)
            with self._current_prefix(f'{iteration:02d}_'):
                self.partial_fit(X, y, **fit_params)

            if self.verbose:
                self._print_weights(f"iteration {iteration}")
            
            previous_tree = copy.deepcopy(self.tree_)
            previous_params = dict(self.get_all_learnable_params())

            valid_score = self.history[-1][scoring]
            improved = valid_score < best_valid_score if ('loss' in checkpoint.monitor) else \
                valid_score > best_valid_score
            if improved:
                best_valid_score = valid_score
            else:
                self.tree_.unsplit_node(index=suboptimal_leaf.index)

            iteration += 1
        
        # Perform global optimization.
        self.set_params(
            callbacks__checkpoint__fn_prefix=f'final_best_'
        )

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
            return y_pred['predictions'][0]
    
    def align_axes(self):
        self.module_._align_axes()
        
    def _collect_all_path_probas(self, iterator):
        all_path_probas = []
        for batch in iterator:
            step = self.validation_step(batch)
            path_probas = step['y_pred']['probas'][1]
            all_path_probas.append(path_probas)
        all_path_probas = list(zip(*all_path_probas))
        return [torch.cat(x) for x in all_path_probas]

    def prune_tree(self, dataset, pruning_threshold=0.05):
        iterator = self.get_iterator(dataset, training=False)
        all_path_probas = self._collect_all_path_probas(iterator)

        eliminate_node = []
        for path_probas_per_layer in all_path_probas:
            average_path_probas = path_probas_per_layer.mean(dim=0)
            eliminate_node.extend(
                (average_path_probas < pruning_threshold).tolist()
            )
        
        self.module_._perform_node_pruning(eliminate_node)
    
    def save_tree(self, features, classes, dataset=None, suffix=''):
        if dataset is not None:
            # @TODO: Visualize patient trajectories.
            iterator = self.get_iterator(dataset, training=False)
            batch = next(iter(iterator))
            step = self.validation_step(batch)
            args = (
                step['y_pred']['probas'][0],
                *step['y_pred']['other']
            )
        else:
            args = ()
        graph = self.module_._draw_tree(features, classes, *args)
        graph.render('tree%s' % suffix, self.results_path, view=False, format='png')


class RDTClassifer(SDTClassifer):
    def __init__(
        self,
        *args,
        delta1=0.001,
        delta2=0.001,
        **kwargs
    ):
        super(RDTClassifer, self).__init__(*args, **kwargs)

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
            name = self.file_name_prefix_ + mode + '_probas'
            plot_save_stats(self.history, probas, self.results_path, name)

            penalties = ['evolution_penalty', 'behavior_penalty', 'splitting_penalty']
            penalties = [f'{mode}_{penalty}' for penalty in penalties]
            name = self.file_name_prefix_ + mode + '_penalties'
            plot_save_stats(self.history, penalties, self.results_path, name)


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
        ccp_alpha=0.0
    ):
        super().__init__(
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
            ccp_alpha=0.0
        )


class DummyClassifier(ClassifierMixin, dummy.DummyClassifier):
    def __init__(
        self, *, strategy='prior', random_state=None, constant=None,
    ):
        super().__init__(
            strategy=strategy,
            random_state=random_state,
            constant=constant
        )


class PrototypeClassifier(BaseClassifier):
    loss_terms = [
        'ce_loss',
        'div_loss',
        'cl_loss',
        'ev_loss'
    ]

    def __init__(
        self, 
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

    def _project_prototypes(self, iterator):
        inputs, similarities, encodings = [], [], []
        
        for batch in iterator:
            self._set_training(False)
            Xi, _ = unpack_data(batch)
            with torch.no_grad():
                _, _similarities, _encodings = self.infer(Xi)
            if isinstance(Xi, PackedSequence):
                Xi = unpack_sequence(Xi)
                Xi = torch.cat(Xi)
            inputs += [Xi]
            similarities += [_similarities]
            encodings += [_encodings]
        
        inputs = torch.cat(inputs)
        similarities = torch.cat(similarities)
        encodings = torch.cat(encodings)
        
        _, max_indices = torch.max(similarities, dim=0)
        projections = encodings[max_indices]
        input_prototypes = inputs[max_indices]
        
        return projections, input_prototypes, max_indices

    def _project_and_record_prototypes(self, dataset_train):
        assert hasattr(self.module_, 'prototype_layer')
        iterator_train = self.get_iterator(dataset_train, training=True)
        projections, input_prototypes, max_indices = \
            self._project_prototypes(iterator_train)
        self.history.record('prototype_indices', max_indices.cpu().tolist())
        self.history.record('prototypes', input_prototypes.cpu().tolist())
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
        if not 'prototypes' in self.history[-1]:
            dataset_train, _ = self.get_split_datasets(
                X, y, **fit_params
            )
            self._project_and_record_prototypes(dataset_train)

        return self

    def on_train_end(self, net, X=None, y=None, **kwargs):
        super(PrototypeClassifier, self).on_train_end(net, X, y, **kwargs)
        for mode in ['train', 'valid']:
            loss_terms = [f'{mode}_{loss_term}' for loss_term in self.loss_terms]
            name = self.file_name_prefix_ + mode + '_penalties'
            plot_save_stats(self.history, loss_terms, self.results_path, name)


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
        return [
            self.estimator_s,
            self.estimator_t
        ]
    
    def _split_data(self, X, y, prev_therapy_index):
        y_prev = np.argmax(X[:, prev_therapy_index], axis=1)
        y_switch = 1 * (y_prev != y)
        switch = y_switch > 0
        return X, y_switch, X[switch], y[switch]
    
    def fit(self, X, y, prev_therapy_index, X_valid=None, y_valid=None):
        """Fit model.
        
        We assume that the labels are encoded into values between
        0 and n_classes-1. We also assume that `prev_therapy_index` 
        contains n_classes elements.
        """

        self._check_fit_args(X, y, prev_therapy_index)
        
        self.prev_therapy_index_ = prev_therapy_index

        Xs, ys, Xt, yt = self._split_data(X, y, prev_therapy_index)
        
        fit_params_s, fit_params_t = {}, {}
        if X_valid is not None and y_valid is not None:
            Xs_valid, ys_valid, Xt_valid, yt_valid = \
                self._split_data(X_valid, y_valid, prev_therapy_index)
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
        assert len(np.unique(y)) == n_classes

        # Check input `X`.
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
