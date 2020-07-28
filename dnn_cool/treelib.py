from treelib import Tree


class TreeLibExplainer:

    def __init__(self, task_flow, prefix=''):
        self.task_flow = task_flow

    def __call__(self, *args, **kwargs):
        tree = Tree()

        return tree