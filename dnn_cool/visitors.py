from dataclasses import dataclass
from typing import Dict


@dataclass
class VisitorData:
    predictions: Dict
    targets: Dict

    def __getattr__(self, item):
        return self

    def visit(self, task, prefix):
        raise NotImplementedError()


class VisitorOut:

    def __iadd__(self, other):
        raise NotImplementedError()

    # Pipeline compatibility
    def __getattr__(self, item):
        return self

    # Pipeline compatibility
    def __invert__(self):
        return self

    # Pipeline compatibility
    def __or__(self, other):
        return self


def get_visitor_data(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, VisitorData):
            return arg


class LeafVisitor:

    def __init__(self, task, prefix):
        self.task = task
        self.prefix = prefix

    def __call__(self, *args, **kwargs):
        visitor_data = get_visitor_data(*args, **kwargs)
        return visitor_data.visit(self.task, self.prefix)


class CompositeVisitor:

    def __init__(self, task_flow, visitor_data_cls, visitor_out_cls, prefix=''):
        self.task_flow = task_flow
        self.prefix = prefix

        self.flow = task_flow.get_flow_func()
        self.prefix = prefix
        self.task_name = task_flow.get_name()
        self.visitor_data_cls = visitor_data_cls
        self.visitor_out_cls = visitor_out_cls

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = LeafVisitor(task, prefix)
            else:
                instance = CompositeVisitor(task, visitor_data_cls, visitor_out_cls, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, data):
        flow_result = self.flow(self, data, self.visitor_out_cls())
        return flow_result


class RootCompositeVisitor:

    def __init__(self, task_flow, visitor_data_cls, visitor_out_cls, prefix=''):
        self.composite_visitor = CompositeVisitor(task_flow, visitor_data_cls, visitor_out_cls, prefix)
        self.visitor_data_cls = visitor_data_cls

    def __call__(self, predictions, targets):
        flow_result = self.composite_visitor(self.visitor_data_cls(predictions, targets))
        return flow_result.data
