from dataclasses import dataclass
from typing import Callable, Union, Tuple

import numpy as np
from treelib import Tree, Node

from dnn_cool.modules import CompositeModuleOutput
from dnn_cool.utils import any_value


def find_results_for_treelib(*args, **kwargs):
    for arg in args:
        if isinstance(arg, Results):
            return arg

    for arg in kwargs.values():
        if isinstance(arg, Results):
            return arg


@dataclass
class Precondition:
    path: str

    def __invert__(self):
        return self


class TreeExplanation:

    def __init__(self, tree: Tree, start_node, results, prefix: str):
        self.tree = tree
        self.start_node = start_node
        self.results = results
        self.precondition = None
        self.prefix = prefix

    def __iadd__(self, other):
        if self.start_node is None:
            return other
        parent = self.start_node.identifier if other.precondition is None else other.precondition.path
        if parent in self.tree:
            self.tree.paste(parent, other.tree)
        return self

    def __getattr__(self, item):
        return Precondition(self.prefix + item)

    def __or__(self, precondition: Precondition):
        self.precondition = precondition
        return self


def _to_print_str(arr):
    if arr.size == 1:
        res = arr.reshape((1,))[0]
        if isinstance(res, float) or isinstance(res, np.floating):
            res = f'{res:.4f}'
        return res
    return arr


def default_leaf_tree_explainer(task_name: str,
                                decoded: np.ndarray,
                                activated: np.ndarray,
                                logits: np.ndarray,
                                node_identifier: str) -> Tuple[Tree, Node]:
    decoded = _to_print_str(decoded)
    activated = _to_print_str(activated)
    logits = _to_print_str(logits)
    description = f'{task_name} | ' \
                  f'decoded: {decoded}, ' \
                  f'activated: {activated}, ' \
                  f'logits: {logits}'
    tree = Tree()
    start_node = tree.create_node(description, node_identifier)
    return tree, start_node


class LeafExplainer:

    def __init__(self, task_name: str, prefix: str, tree_func=default_leaf_tree_explainer):
        self.prefix = prefix
        self.task_name = task_name
        self.tree_func = tree_func

    def __call__(self, *args, **kwargs) -> TreeExplanation:
        results: Results = find_results_for_treelib(*args, **kwargs)
        path = self.prefix + self.task_name

        decoded = results.module_output.decoded[path][results.idx].detach().cpu().numpy()
        activated = results.module_output.activated[path][results.idx].detach().cpu().numpy()
        logits = results.module_output.logits[path][results.idx].detach().cpu().numpy()
        precondition = results.module_output.preconditions.get(path)
        if precondition is not None:
            precondition = precondition[results.idx][0].item()

        should_create_node = (precondition is None) or (precondition is True)
        tree, start_node = Tree(), None
        if should_create_node:
            tree, start_node = self.tree_func(self.task_name, decoded, activated, logits, f'inp_{results.idx}.{path}')
        return TreeExplanation(tree, start_node, results, self.prefix)


class Results:

    def __init__(self, module_output, idx=None):
        self.module_output = module_output
        self.idx = idx

    # Pipeline compatibility
    def __getattr__(self, item):
        return self


class TreeExplainer:

    def __init__(self, task_name: str, flow_func: Callable, flow_tasks, prefix=''):
        self.task_name = task_name
        self.flow = flow_func
        self.prefix = prefix

        for key, task in flow_tasks.items():
            if not task.has_children():
                instance = LeafExplainer(task.get_name(), prefix=prefix, tree_func=task.get_treelib_explainer())
            else:
                instance = TreeExplainer(task.get_name(),
                                         task.get_flow_func(),
                                         task.tasks,
                                         prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, x: Union[CompositeModuleOutput, Results]) -> Union[Tree, TreeExplanation]:
        if not isinstance(x, Results):
            x = Results(x)

        if x.idx is None:
            n = len(any_value(x.module_output.logits))
            tree = Tree()
            batch_node = tree.create_node(tag='batch', identifier='batch')
            for i in range(n):
                inp_tree = Tree()
                inp_node = inp_tree.create_node(tag=f'inp {i}', identifier=f'inp_{i}.{self.prefix}')
                x_for_id = Results(x.module_output, i)
                out = TreeExplanation(inp_tree, inp_node, x_for_id, f'inp_{i}.')
                out = self.flow(self, x_for_id, out)
                tree.paste(batch_node.identifier, out.tree)

            if len(self.prefix) == 0:
                return tree
            return TreeExplanation(tree, start_node=batch_node, results=x, prefix=self.prefix)

        tree = Tree()
        start_node = tree.create_node(tag=self.task_name, identifier=f'inp_{x.idx}.{self.prefix}.{self.task_name}')
        out = TreeExplanation(tree, start_node=start_node, results=x, prefix=f'inp_{x.idx}.{self.prefix}')
        out = self.flow(self, x, out)

        no_new_nodes_added = (len(out.tree.nodes) == 1) and (out.start_node.identifier in out.tree)
        # if all the nodes of a nested flow are empty, remove the whole flow node.
        if no_new_nodes_added:
            out.tree = Tree()
        if len(self.prefix) == 0:
            return out.tree
        return out
