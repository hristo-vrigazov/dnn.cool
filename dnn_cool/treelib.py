from dataclasses import dataclass, field
from typing import List

from treelib import Tree

from dnn_cool.utils import any_value


def find_results_for_treelib(*args, **kwargs):
    for arg in args:
        if isinstance(arg, ResultsForTreelib):
            return arg

    for arg in kwargs.values():
        if isinstance(arg, ResultsForTreelib):
            return arg


class TreeLibExplainer:

    def __init__(self, task, prefix):
        self.prefix = prefix
        self.task_name = task.get_name()

    def __call__(self, *args, **kwargs):
        results: ResultsForTreelib = find_results_for_treelib(*args, **kwargs)
        path = self.prefix + self.task_name

        decoded = results.module_output.decoded[path][results.idx].detach().cpu().numpy()
        activated = results.module_output.activated[path][results.idx].detach().cpu().numpy()
        logits = results.module_output.logits[path][results.idx].detach().cpu().numpy()

        description = f'{path} | decoded: {decoded}, activated: {activated}, logits: {logits}'

        tree = Tree()
        tree.create_node(description, path)
        return tree


class ResultsForTreelib:

    def __init__(self, module_output, idx=None):
        self.module_output = module_output
        self.idx = idx

    # Pipeline compatibility
    def __getattr__(self, item):
        return self


class TreeExplanation:

    def __init__(self, start_node):
        self.tree = Tree()
        self.start_node = start_node
        self.tree.add_node(start_node)

    def __iadd__(self, other):
        self.tree.paste(self.start_node.identifier, other)
        self.tree.show()


class CompositeTreeLibExplainer:

    def __init__(self, task_flow, prefix=''):
        self.task_flow = task_flow

        self.flow = task_flow.get_flow_func()
        self.prefix = prefix

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = TreeLibExplainer(task, prefix)
            else:
                instance = CompositeTreeLibExplainer(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, x):
        if not isinstance(x, ResultsForTreelib):
            x = ResultsForTreelib(x)

        if x.idx is None:
            n = len(any_value(x.module_output.logits))
            tree = Tree()
            batch_node = tree.create_node(tag='batch', identifier='batch')
            for i in range(n):
                inp_node = tree.create_node(tag=f'inp {i}', identifier=f'inp_{i}', parent=batch_node)
                x_for_id = ResultsForTreelib(x.module_output, i)
                out = TreeExplanation(inp_node)
                out = self.flow(self, x_for_id, out)
                sub_tree = out.reduce()
                tree.paste(batch_node, sub_tree)
            return tree

        out = TreeExplanation()
        out = self.flow(self, x, out)
        return out.reduce()
