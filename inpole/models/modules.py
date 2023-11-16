import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, unpack_sequence

import numpy as np
from scipy.special import softmax
from skorch.utils import params_for

from ..utils import compute_squared_distances
from ..tree import get_node_indices_per_layer, draw_tree


ALL_NODE_MODULES = [
    'inner_nodes',
    'leaf_nodes',
    'history_leaf_nodes',
    'observation_leaf_nodes'
]


def _concatenate_and_unpack(data_sequence, *args):
    data = torch.cat(data_sequence)
    packed_sequence = PackedSequence(data, *args)
    sequences = unpack_sequence(packed_sequence)
    return torch.cat(sequences)


class LinearFromWeights(nn.Linear):
    def __init__(self, weights):
        super(LinearFromWeights, self).__init__(
            in_features=weights.shape[1],
            out_features=weights.shape[0],
            bias=False,
            device=weights.device,
            dtype=weights.dtype
        )
        with torch.no_grad():
            self.weight.copy_(weights)


class UnidimensionalInnerNodes(nn.Module):
    def __init__(self, weight, input_dim, hidden_dim=0):
        super(UnidimensionalInnerNodes, self).__init__()
        self.weight = weight
        self._split_sizes = [1, hidden_dim, input_dim]
    
    def forward(self, x):
        # Split the weights.
        w = torch.t(self.weight)
        b, wh, wz = torch.split(w, self._split_sizes, dim=0)

        # Split the input.
        _, h, z = torch.split(x, self._split_sizes, dim=1)
        
        # Marginalize out the history into the bias term.
        #
        # If there is no hidden dimension, torch.mm(h, wh) returns
        # a zero tensor of shape (batch size, # inner nodes).
        bh = b + torch.mm(h, wh)  # Shape (batch size, # inner nodes)

        # Select the largest weight for each inner node.
        wmax, max_indices = torch.max(wz, dim=0)  # Shape (# inner nodes,)
        
        # Select the observations corresponding to the largest weights.
        zmax = z[:, max_indices]  # Shape (batch size, # inner nodes)

        # Compute the thresholds.
        thresholds = -(bh / wmax)

        # For each sample in the batch, compute
        # zmax > - bh / wmax if wmax > 0
        # zmax < - bh / wmax if wmax < 0
        condition = (wmax > 0).unsqueeze(0).expand_as(zmax)
        input = zmax > thresholds
        other = zmax < thresholds
        return (
            torch.where(condition, input, other).type(torch.float32),
            zmax,
            thresholds
        )


class SDT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        tree,
        prediction='max'
    ):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.tree = tree
        self.prediction = prediction

        self._validate_parameters()

        self.penalty_decay = [2 ** (-d) for d in range(tree.depth)]

        num_inner_nodes = len(tree.inner_nodes)
        num_leaf_nodes = len(tree.leaf_nodes)

        self.inner_nodes = nn.Sequential(
            nn.Linear(
                self.input_dim + 1,  # Add bias
                num_inner_nodes,
                bias=False
            ),
            nn.Sigmoid()
        )

        self.leaf_nodes = nn.Sequential(
            nn.Linear(
                num_leaf_nodes,
                self.output_dim,
                bias=False
            ),
            nn.Identity()
        )

    def forward(self, X, predict_only=False):
        path_probas, penalty, all_path_probas, other = self._forward(X)

        if self.prediction == 'max':
            masked_path_probas = self._mask_path_probas(path_probas)
            logits = self.leaf_nodes(masked_path_probas)
        elif self.prediction == 'mean':
            logits = self.leaf_nodes(path_probas)

        if predict_only:
            return {'predictions': (logits,)}
        else:
            return {
                'probas': (path_probas, all_path_probas),
                'penalties': (penalty,),
                'predictions': (logits,),
                'other': other
            }

    def _forward(self, X):
        device = X.device
        batch_size = X.size()[0]
        X = self._data_augment(X)

        if isinstance(self.inner_nodes, UnidimensionalInnerNodes):
            path_probas, zmax, thresholds = self.inner_nodes(X)
        else:
            path_probas = self.inner_nodes(X)
            thresholds = torch.full_like(path_probas.detach(), np.nan)
            zmax = thresholds.clone()
        
        path_probas = torch.unsqueeze(path_probas, dim=2)
        path_probas = torch.cat((1 - path_probas, path_probas), dim=2)

        mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        penalty = torch.tensor(0.0).to(device)

        all_path_probas = [mu.view(batch_size, -1)]
        
        full_node_indices_per_layer = get_node_indices_per_layer(self.tree.depth)
        
        # Propagate through the tree to compute the final path probabilities
        # and the regularization term.
        for i_layer in range(self.tree.depth):
            full_node_indices = full_node_indices_per_layer[i_layer]

            # Get each node's index in the current layer.
            layer_aligned_node_indices = \
                self.tree.layer_aligned_inner_node_indices_per_layer[i_layer]

            # Get each node's index in the entire tree.
            tree_aligned_node_indices = \
                self.tree.tree_aligned_inner_node_indices_per_layer[i_layer]
            
            num_nodes_in_full_layer = len(full_node_indices)
            num_nodes_in_layer = len(layer_aligned_node_indices)
    
            if num_nodes_in_full_layer > num_nodes_in_layer:
                _path_probas = torch.full(
                    (batch_size, num_nodes_in_full_layer, 2),
                    1.0,
                    dtype=_path_probas.dtype,
                    device=_path_probas.device,
                )
                _path_probas[:, layer_aligned_node_indices, :] = \
                    path_probas[:, tree_aligned_node_indices, :]
            else:
                _path_probas = path_probas[:, full_node_indices, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term.
            penalty += self._cal_penalty(i_layer, mu, _path_probas)
 
            # `mu` grows from the initial shape (B, 1, 1) to (B, 1, 2) to (B, 2, 2)
            # to (B, 4, 2) to (B, 8, 2), etc.
            #
            # For example, when `layer_idx = 1`, the line below does the following transformation:
            # mu = [[[a, b]], [[a, b]]]  (shape (B, 1, 2))
            # mu.view(batch_size, -1, 1) = [[[a], [b]], [[a], [b]]]  (shape (B, 2, 1))
            # mu.view(batch_size, -1, 1).repeat(1, 1, 2) = [[[a, a], [b, b]], [[a, a], [b, b]]]  (shape (B, 2, 2))
            mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            mu = mu * _path_probas

            _all_path_probas = mu.detach().clone()
            mask = torch.zeros(num_nodes_in_full_layer, dtype=torch.bool, device=mu.device)
            mask[layer_aligned_node_indices] = True
            _all_path_probas[:, ~mask, :] = np.nan
            _all_path_probas = _all_path_probas.view(batch_size, -1)
            all_path_probas += [_all_path_probas]
    
        mu = mu.view(batch_size, -1)
        
        if mu.shape[1] > len(self.tree.leaf_nodes):
            mu = mu[:, self.tree.bottom_layer_aligned_leaf_node_indices]

        return mu, penalty, all_path_probas, (zmax, thresholds)
        
    def _cal_penalty(self, i_layer, _mu, _path_prob):
        num_nodes_in_layer = 2 ** i_layer
        num_nodes_in_next_layer = 2 ** (i_layer + 1)

        penalty = torch.tensor(0.0).to(_mu.device)

        batch_size = _mu.shape[0]
        _mu = _mu.view(batch_size, num_nodes_in_layer)
        _path_prob = _path_prob.view(batch_size, num_nodes_in_next_layer)

        for i in range(num_nodes_in_layer):
            node_index = 2 ** i_layer - 1 + i
            if not node_index in self.tree.inner_nodes:
                continue
            
            # Find the tree index of the right child.
            right_child_index = 2 * node_index + 2

            # Convert the tree index to an index in 
            # [0, `num_nodes_in_next_layer`).
            i_rc = right_child_index - num_nodes_in_next_layer + 1

            alpha = torch.sum(_path_prob[:, i_rc] * _mu[:, i], dim=0)
            alpha /= torch.sum(_mu[:, i], dim=0)
            _penalty = (torch.log(alpha) + torch.log(1 - alpha))

            if _penalty.isnan() or _penalty.isinf():
                # @TODO: Handle this case.
                continue

            penalty -= 0.5 * self.penalty_decay[i_layer] * _penalty

        return penalty
    
    def _data_augment(self, X):
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(X.device)
        X = torch.cat((bias, X), 1)
        return X

    def _validate_parameters(self):
        if not self.prediction in ['mean', 'max']:
            raise ValueError(
                f"Invalid value '{self.prediction}' for parameter `prediction`. "
                "Valid options are: 'mean', 'max'."
            )
    
    def _mask_path_probas(self, path_probas):
        path_probas_np = path_probas.detach().cpu().numpy()
        assert np.allclose(path_probas_np.sum(axis=1), 1.0)
        max_indices = path_probas.argmax(dim=1).unsqueeze(1)
        mask = torch.zeros_like(path_probas)
        mask.scatter_(1, max_indices, 1.0)
        return mask

    def _align_axes(self):
        if not isinstance(self.inner_nodes, UnidimensionalInnerNodes):
            self.inner_nodes = UnidimensionalInnerNodes(
                self.inner_nodes[0].weight, self.input_dim,
            )
    
    def _update_weights_after_pruning(
        self,
        removed_inner_node_indices,
        removed_leaf_node_indices
    ):
        for name, module in self.named_modules():
            if name in ALL_NODE_MODULES:
                if isinstance(module, nn.Sequential):
                    weight = module[0].weight.detach()
                else:
                    weight = module.weight.detach()
                
                if name == 'inner_nodes':
                    mask = np.ones(weight.shape[0], dtype=bool)
                    mask[removed_inner_node_indices] = False
                    new_weight = weight[mask]
                else:
                    assert 'leaf_nodes' in name
                    mask = np.ones(weight.shape[1], dtype=bool)
                    mask[removed_leaf_node_indices] = False
                    new_weight = weight[:, mask]
                
                if isinstance(module, UnidimensionalInnerNodes):
                    module.weight = nn.Parameter(new_weight)
                else:
                    assert isinstance(module, nn.Sequential)
                    module[0] = LinearFromWeights(new_weight)
    
    def _perform_node_pruning(self, eliminate_node):
        node_indices_to_remove = np.flatnonzero(eliminate_node)

        # @TODO: Handle this case although it should not happen in practice.
        for node in self.tree.inner_nodes.values():
            if node.index in node_indices_to_remove:
                continue
            num_removed_children = 0
            for child in node.children:
                if child.index in node_indices_to_remove:
                    num_removed_children += 1
            if num_removed_children == 2:
                raise ValueError(
                    f"Inner node {node.index} is childless after pruning!"
                )

        tree_aligned_removed_inner_node_indices = []
        tree_aligned_removed_leaf_node_indices = []
        for index in node_indices_to_remove:
            node = self.tree.nodes[index]
            if node.is_leaf:
                tree_aligned_removed_leaf_node_indices.append(
                    self.tree.leaf_node_index(node)
                )
            else:
                tree_aligned_removed_inner_node_indices.append(
                    self.tree.inner_node_index(node)
                )

        sorted_nodes = list(self.tree.sorted_nodes.values())
        for node in sorted_nodes[::-1]:  # From leaves to root
            if node.index in node_indices_to_remove:
                self.tree.remove_leaf_node(node.index)
        
        # Remove single child nodes.
        tree_aligned_removed_inner_node_indices.extend(
            self.tree.remove_single_child_nodes()
        )

        # Update weights.
        self._update_weights_after_pruning(
            tree_aligned_removed_inner_node_indices,
            tree_aligned_removed_leaf_node_indices
        )
        
    def _draw_tree(self, features, classes, edge_attrs=None):
        assert isinstance(self.inner_nodes, UnidimensionalInnerNodes)
    
        leaf_node_weights = self.leaf_nodes[0].weight.detach().cpu().numpy()
        leaf_node_probas = softmax(leaf_node_weights, axis=0)
        leaf_node_preds = np.argmax(leaf_node_weights, axis=0).astype(int)
        
        w = torch.t(self.inner_nodes.weight)
        _, _, wz = torch.split(w, self.inner_nodes._split_sizes, dim=0)
        _, max_indices = torch.max(wz, dim=0)

        inner_node_labels = {}
        leaf_node_labels = {}
        for node in self.tree.nodes.values():
            if node.is_leaf:
                index = self.tree.leaf_node_index(node)
                pred_class = leaf_node_preds[index]
                class_proba = leaf_node_probas[pred_class, index]
                label = f'{classes[pred_class]} ({class_proba:.2f})'
                leaf_node_labels[node.index] = label
            else:
                index = self.tree.inner_node_index(node)
                max_feature = max_indices[index]
                label = features[max_feature]
                inner_node_labels[node.index] = label
            
        return draw_tree(self.tree, inner_node_labels, leaf_node_labels, edge_attrs)


class RDT(SDT):
    def __init__(
        self,
        hidden_dim,  # M: history dimensionality
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.hidden_dim = hidden_dim

        num_inner_nodes = len(self.tree.inner_nodes)
        num_leaf_nodes = len(self.tree.leaf_nodes)

        # Redefine inner nodes.
        self.inner_nodes = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.input_dim + 1,
                num_inner_nodes,
                bias=False
            ),
            nn.Sigmoid()
        )

        # Redefine action leaf nodes.
        self.leaf_nodes = nn.Sequential(
            nn.Linear(
                num_leaf_nodes,
                self.output_dim,
                bias=False
            ),
            nn.Identity()
        )

        # Add history leaf nodes.
        self.history_leaf_nodes = nn.Sequential(
            nn.Linear(
                num_leaf_nodes,
                self.hidden_dim,
                bias=False
            ),
            nn.Tanh()
        )

        # Add observation leaf nodes.
        self.observation_leaf_nodes = nn.Sequential(
            nn.Linear(
                num_leaf_nodes,
                self.input_dim,
                bias=False,
            ),
            nn.Tanh()
        )

    def forward(self, input, h_0=None, predict_only=False):
        assert isinstance(input, PackedSequence)
        
        observations, batch_sizes, sorted_indices, unsorted_indices = input

        if h_0 is None:
            max_batch_size = int(batch_sizes[0])
            history = torch.zeros(
                max_batch_size,
                self.hidden_dim,
                dtype=observations.dtype,
                device=observations.device
            )
        
        path_probas_per_step = []
        all_path_probas_per_step = []
        logits_per_step = []
        histories_per_step = []
        observations_per_step = []
        zmax_per_step = []
        thresholds_per_step = []

        device = observations.device
        reg1 = torch.tensor(0.0).to(device)  # Fidelity to true evolution
        reg2 = torch.tensor(0.0).to(device)  # Fidelity to demonstrated behavior
        reg3 = torch.tensor(0.0).to(device)  # Splitting penalty

        pred_observation = None
        pred_logits = None
        
        offset = 0
        last_batch_size = batch_sizes[0]

        for batch_size in batch_sizes:  # Loop over time steps
            if last_batch_size - batch_size > 0:
                history = history[:batch_size]
                pred_observation = pred_observation[:batch_size]
                if not predict_only:
                    pred_logits = pred_logits[:batch_size]
            
            observation = observations[offset:offset+batch_size]

            offset += batch_size
            last_batch_size = batch_size

            if pred_observation is not None:  # Ignore first time step
                reg1 += F.mse_loss(
                    observation,
                    pred_observation
                )

            input = torch.cat([history, observation], dim=1)
            path_probas, penalty, all_path_probas, (zmax, thresholds) = \
                self._forward(input)
            
            path_probas_per_step.append(path_probas)
            all_path_probas_per_step.append(all_path_probas)
            zmax_per_step.append(zmax)
            thresholds_per_step.append(thresholds)

            if self.prediction == 'max':
                path_probas = self._mask_path_probas(path_probas)
            logits = self.leaf_nodes(path_probas)
            history = self.history_leaf_nodes(path_probas)  # Next history
            pred_observation = self.observation_leaf_nodes(path_probas)  # Predicted next observation

            logits_per_step.append(logits)
            histories_per_step.append(history)
            observations_per_step.append(pred_observation)

            if pred_logits is not None:  # Ignore first step
                reg2 += F.kl_div(
                    F.log_softmax(pred_logits, dim=1),
                    F.log_softmax(logits, dim=1),
                    reduction='batchmean',
                    log_target=True
                )
            
            reg3 += penalty

            # To compute the loss, we need the predictive distribution 
            # over classes computed with estimated observations.
            if not predict_only:
                # Concatenate the next history and the predicted next observation.
                next_input = torch.cat([history, pred_observation], dim=1)
                next_path_probas, _, _, _ = self._forward(next_input)
                if self.prediction == 'max':
                    next_path_probas = self._mask_path_probas(next_path_probas)
                pred_logits = self.leaf_nodes(next_path_probas)
        
        all_path_probas_per_step = list(zip(*all_path_probas_per_step))
        
        args = (batch_sizes, sorted_indices, unsorted_indices)
        path_probas = _concatenate_and_unpack(path_probas_per_step, *args)
        all_path_probas = [
            _concatenate_and_unpack(x, *args) for x in all_path_probas_per_step
        ]
        logits =  _concatenate_and_unpack(logits_per_step, *args)
        histories =  _concatenate_and_unpack(histories_per_step, *args)
        observations = _concatenate_and_unpack(observations_per_step, *args)
        zmax = _concatenate_and_unpack(zmax_per_step, *args)
        thresholds = _concatenate_and_unpack(thresholds_per_step, *args)

        probas = (path_probas, all_path_probas)
        penalties = (reg1, reg2, reg3)
        predictions = (logits, histories, observations)

        if predict_only:
            return {'predictions': predictions}
        else:
            return {
                'probas': probas,
                'penalties': penalties,
                'predictions': predictions,
                'other': (zmax, thresholds)
            }

    def _align_axes(self):
        self.inner_nodes = UnidimensionalInnerNodes(
            self.inner_nodes[0].weight,
            self.input_dim,
            self.hidden_dim
        )


class NNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=64,
        hidden_dims=(),
        nonlinearity=nn.ReLU()
    ):
        super(NNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.nonlinearity = nonlinearity

        sizes = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList([])
        for i in range(len(sizes) - 1):
            in_features = sizes[i]
            out_features = sizes[i+1]
            self.layers += [nn.Linear(in_features, out_features)]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = self.nonlinearity(layer(inputs))
        return inputs


class RNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=64,
        num_layers=1,
        nonlinearity='tanh'
    ):
        super(RNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_dim,
            output_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True
        )

    def forward(self, inputs):
        assert isinstance(inputs, PackedSequence)
        encodings, _ = self.rnn(inputs)
        encodings = unpack_sequence(encodings)
        return torch.cat(encodings)


class Prototype(nn.Module):
    def __init__(self, input_dim, output_dim, gamma=1.):
        super(Prototype, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        
        # Initialize weights from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k}))`
        # where :math:`k = \frac{1}{\text{input\_dim}}`.
        prototypes = torch.rand(output_dim, input_dim)
        k = 1 / math.sqrt(input_dim)
        self.prototypes = nn.Parameter(2*k*prototypes - k)
    
    def forward(self, inputs):
        squared_distances = compute_squared_distances(inputs, self.prototypes)
        similarities = torch.exp(torch.neg(squared_distances) / (self.gamma*self.gamma))
        return similarities
    
    def extra_repr(self):
        return "input_dim={}, output_dim={}".format(self.input_dim, self.output_dim)


class PrototypeNetwork(nn.Module):
    def __init__(self, input_dim, encoder, num_prototypes, output_dim, **kwargs):
        super(PrototypeNetwork, self).__init__()
        
        encoder_params = params_for('encoder', kwargs)
        if not 'input_dim' in encoder_params:
            encoder_params['input_dim'] = input_dim
        self.encoder = encoder(**encoder_params)

        self.hidden_size = self.encoder.output_dim
        self.num_prototypes = num_prototypes
        self.output_dim = output_dim
        
        self._use_prototypes = (num_prototypes > 0)

        if self._use_prototypes:
            self.prototype_layer = Prototype(self.hidden_size, num_prototypes)
            self.output_layer = nn.Linear(num_prototypes, output_dim)
        else:
            self.output_layer = nn.Linear(self.hidden_size, output_dim)

    def forward(self, inputs):
        encodings = self.encoder(inputs)
        if self._use_prototypes:
            similarities = self.prototype_layer(encodings)
            outputs = self.output_layer(similarities)
        else:
            similarities = torch.Tensor([0])  # Instead of `None`
            outputs = self.output_layer(encodings)
        return outputs, similarities, encodings

    def set_prototypes(self, new_prototypes):
        if self._use_prototypes:
            with torch.no_grad():
                self.prototype_layer.prototypes.copy_(new_prototypes)
