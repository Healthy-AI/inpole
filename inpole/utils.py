import os
import math
import copy
import torch
import datetime
import graphviz
from pathlib import Path


def create_decision_tree(depth):
    tree = create_decision_stump()
    while (
        tree.depth < depth or
        len(tree.inner_nodes) < 2 ** depth - 1
    ):
        leaf_node_indices = list(tree.leaf_nodes)
        leaf_node_indices.sort()  # Sort from smallest to largest
        tree = tree.split_node(index=leaf_node_indices[0])
    return tree


def create_decision_stump():
    root = Node(index=0)
    stump = Tree(nodes=[root])
    return stump.split_node(index=0)


class Tree:
    def __init__(self, nodes=[]):
        self.nodes = {
            node.index: node for node in nodes
        }
        self._update()

    def split_node(self, index):
        node = self.nodes[index]
        assert node.index == index
        node.add_children()
        for child in node.children:
            self.nodes[child.index] = child
        self._update()
        return self

    def unsplit_node(self, index):
        node = self.nodes[index]
        assert node.index == index
        for child in node.children:
            self.nodes.pop(child.index)
        node.remove_children()
        node._is_suboptimal = False
        self._update()
        return self

    def remove_leaf_node(self, index):
        node = self.nodes[index]
        assert node.index == index
        assert node.is_leaf
        self.nodes.pop(index)
        node.parent.remove_child(node)
        self._update()
        return self

    def _update_index_of_descendants(self, node):
        if node.left_child is not None:
            node.left_child.index = 2 * node.index + 1
            self._update_index_of_descendants(node.left_child)
        if node.right_child is not None:
            node.right_child.index = 2 * node.index + 2
            self._update_index_of_descendants(node.right_child)
        return self

    def remove_single_child_nodes(self):
        # Check which nodes are to be removed.
        removed_inner_node_indices = []
        for node in self.nodes.values():
            if len(node.children) == 1:
                assert not node.is_leaf
                removed_inner_node_indices.append(
                    self.inner_node_index(node)
                )

        # Remove nodes and update indices and connections.
        sorted_nodes = list(self.sorted_nodes.items())
        for index, node in sorted_nodes[::-1]:  # Iterate from leaves to root
            children = node.children
            
            if len(children) == 1:
                child = children.pop()
                parent = node.parent

                # Connect child and parent.
                child.parent = parent
                if parent and parent.left_child == node:
                    parent.left_child = child
                elif parent and parent.right_child == node:
                    parent.right_child = child

                # Update index of child and its descendants.
                child.index = node.index
                self._update_index_of_descendants(child)
                
                # Remove node.
                self.nodes.pop(index)
        
        self.nodes = {
            node.index: node for node in self.nodes.values()
        }
        self._update()
        
        return removed_inner_node_indices
    
    def inner_node_index(self, node):
        assert not node.is_leaf
        return list(self.sorted_inner_nodes).index(node.index)

    def leaf_node_index(self, node):
        assert node.is_leaf
        return list(self.sorted_leaf_nodes).index(node.index)

    def get_sorted_inner_nodes_in_layer(self, layer_index):
        out = {}
        for index, node in self.sorted_inner_nodes.items():
            if node.layer_index == layer_index:
                out[index] = node
        return out
    
    @property
    def suboptimal_leaves(self):
        leaf_nodes = list(self.leaf_nodes.values())
        suboptimal_leaves = [
            leaf for leaf in leaf_nodes if leaf.is_suboptimal
        ]
        # We want to sort from largest to smallest index; note that
        # `self.sorted_leaf_nodes` contains the nodes sorted from
        # left to right.
        suboptimal_leaves.sort(key=lambda leaf: leaf.index, reverse=True)
        return suboptimal_leaves

    @property
    def inner_nodes(self):
        return self.sorted_inner_nodes
    
    @property
    def leaf_nodes(self):
        return self.sorted_leaf_nodes

    def _update_sorted_nodes(self):
        self.sorted_nodes = dict(sorted(self.nodes.items()))
    
    def _update_sorted_inner_nodes(self):
        self.sorted_inner_nodes = {
            index: node for index, node in self.sorted_nodes.items() if not node.is_leaf
        }
    
    def _update_depth(self):
        if len(self.inner_nodes) == 0:
            self.depth = 0
        else:
            self.depth = max(
                [node.layer_index for node in self.inner_nodes.values()]
            ) + 1
        
    def _update_sorted_leaf_nodes(self):
        def sorter(item):
            index, node = item
            if node.layer_index == self.depth:
                return index
            else:
                # Propagate to the bottom of the tree and return the
                # left-most descendant of the node.
                return get_leaf_node_indices(self.depth, index)[0]
        leaf_nodes = {
            index: node for index, node in self.nodes.items() if node.is_leaf
        }
        self.sorted_leaf_nodes = dict(sorted(leaf_nodes.items(), key=sorter))
    
    def _update(self):
        self._update_sorted_nodes()
        self._update_sorted_inner_nodes()  # Requires `self.sorted_nodes`
        self._update_depth()  # Requires `self.sorted_inner_nodes`
        self._update_sorted_leaf_nodes()  # Requires `self.depth`
    
        self.tree_aligned_inner_node_indices_per_layer = {}
        self.layer_aligned_inner_node_indices_per_layer = {}

        full_inner_node_indices_per_layer = \
            get_node_indices_per_layer(self.depth)

        for i_layer in range(self.depth):
            full_inner_node_indices = full_inner_node_indices_per_layer[i_layer]
            inner_nodes = self.get_sorted_inner_nodes_in_layer(i_layer)
            self.tree_aligned_inner_node_indices_per_layer[i_layer] = [
                self.inner_node_index(node) for node in inner_nodes.values()
            ]
            self.layer_aligned_inner_node_indices_per_layer[i_layer] = [
                index - full_inner_node_indices[0] for index in inner_nodes.keys()
            ]
        
        self.bottom_layer_aligned_leaf_node_indices = [ 
            get_leaf_node_indices(self.depth, index)[0]  # Select the index of the left-most leaf descendant
            if node.layer_index < self.depth else index
            for index, node in self.leaf_nodes.items()
        ]
        self.bottom_layer_aligned_leaf_node_indices.sort()
        # The index of the left-most leaf node always comes first
        # so we can shift the indices accordingly.
        shift = self.bottom_layer_aligned_leaf_node_indices[0]
        self.bottom_layer_aligned_leaf_node_indices = \
            [i - shift for i in self.bottom_layer_aligned_leaf_node_indices]


class Node:
    def __init__(
        self,
        index,
        parent=None,
        left_child=None,
        right_child=None
    ):
        self.index = index
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

    def remove_child(self, child):
        if child == self.left_child:
            self.left_child = None
        elif child == self.right_child:
            self.right_child = None
        else:
            raise ValueError(
                f"Child node with index {child.index} is not a child "
                f"of node with index {self.index}."
            )

    def remove_children(self):
        self.left_child = None
        self.right_child = None
    
    def add_children(self):
        left_child = Node(
            index=2 * self.index + 1,
            parent=self,
        )
        right_child = Node(
            index=2 * self.index + 2,
            parent=self,
        )
        self.left_child = left_child
        self.right_child = right_child
    
    @property
    def layer_index(self):
        return int(math.log2(self.index + 1))

    @property
    def is_suboptimal(self):
        return getattr(
            self, '_is_suboptimal', self.is_leaf
        )
    
    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def children(self):
        return [child for child in [self.left_child, self.right_child] if child is not None]


def get_node_indices_per_layer(depth):
    node_indices_per_layer = {}
    i_start, i_stop = 0, 1
    for i_layer in range(depth):
        node_indices_per_layer[i_layer] = \
            list(range(i_start, i_stop))
        i_start = i_stop
        i_stop = i_start + 2 ** (i_layer + 1)
    return node_indices_per_layer


def get_leaf_node_indices(depth, index):
    result = []
    
    num_inner_nodes = 2 ** depth - 1
    num_leaf_nodes = 2 ** depth
    num_total_nodes = num_inner_nodes + num_leaf_nodes

    if index >= num_inner_nodes:
        return result

    def traverse(node_index, current_depth):
        if current_depth == (depth + 1):
            result.append(node_index)
        else:
            left_child_index = 2 * node_index + 1
            right_child_index = 2 * node_index + 2
            if left_child_index < num_total_nodes:
                traverse(left_child_index, current_depth + 1)
            if right_child_index < num_total_nodes:
                traverse(right_child_index, current_depth + 1)
    
    current_depth = int(math.log2(index + 1)) + 1

    traverse(index, current_depth)

    return result


def draw_tree(tree, inner_node_labels=None, leaf_node_labels=None):
    graph = graphviz.Digraph()

    for node in tree.nodes.values():
        if node.is_leaf:
            if leaf_node_labels:
                label = leaf_node_labels[node.index]
            else:
                label = str(node.index)
            shape = 'circle'
        else:
            if inner_node_labels:
                label = inner_node_labels[node.index]
            else:
                label = str(node.index)
            shape = 'box'
        #label += ' (%s)' % node.index
        graph.node(str(node.index), label, shape=shape)
        if node.parent is not None:
            label = 'Y' if node.parent.right_child == node else 'N'
            graph.edge(str(node.parent.index), str(node.index), label=label)
        
    return graph


def create_results_dir_from_config(
    config,
    suffix=None,
    update_config=False
):
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if suffix is not None:
        time_stamp += '_' + suffix
    results_path = os.path.join(config['results']['path'], time_stamp)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    if update_config:
        config = copy.deepcopy(config)
        config['results']['path'] = results_path
        return results_path, config
    else:
        return results_path


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def compute_squared_distances(x1, x2):
    """Compute squared distances using quadratic expansion.
    
    Reference: https://github.com/pytorch/pytorch/pull/25799.
    """
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    
    x1 = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2 = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    
    return x1.matmul(x2.transpose(-2, -1))
