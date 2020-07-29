from dataclasses import dataclass
from treelib import Tree

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

    def __init__(self, tree, start_node, results, prefix):
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


class LeafExplainer:

    def __init__(self, task, prefix):
        self.prefix = prefix
        self.task_name = task.get_name()

    def __call__(self, *args, **kwargs):
        results: Results = find_results_for_treelib(*args, **kwargs)
        path = self.prefix + self.task_name

        decoded = results.module_output.decoded[path][results.idx].detach().cpu().numpy()
        activated = results.module_output.activated[path][results.idx].detach().cpu().numpy()
        logits = results.module_output.logits[path][results.idx].detach().cpu().numpy()
        precondition = results.module_output.preconditions.get(path)
        if precondition is not None:
            precondition = precondition[results.idx][0].item()

        tree = Tree()
        should_create_node = (precondition is None) or (precondition is True)
        if should_create_node:
            description = f'{self.task_name} | decoded: {decoded}, activated: {activated}, logits: {logits}'
            start_node = tree.create_node(description, f'inp_{results.idx}.{path}')
        else:
            start_node = None
        return TreeExplanation(tree, start_node, results, self.prefix)


class Results:

    def __init__(self, module_output, idx=None):
        self.module_output = module_output
        self.idx = idx

    # Pipeline compatibility
    def __getattr__(self, item):
        return self


class TreeExplainer:

    def __init__(self, task_flow, prefix=''):
        self.task_flow = task_flow

        self.flow = task_flow.get_flow_func()
        self.prefix = prefix
        self.task_name = task_flow.get_name()

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = LeafExplainer(task, prefix)
            else:
                instance = TreeExplainer(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, x):
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
