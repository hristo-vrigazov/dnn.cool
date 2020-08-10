import torch

from dataclasses import dataclass
from typing import Dict


@dataclass
class VisitorData:
    predictions: Dict
    targets: Dict

    def __getattr__(self, item):
        return self


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

    def reduce(self):
        raise NotImplementedError()


def get_visitor_data(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, VisitorData):
            return arg


class LeafVisitor:

    def __init__(self, task, prefix):
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()
        self.prefix = prefix
        self.path = self.prefix + task.get_name()
        self.available = task.get_available_func()

    def __call__(self, *args, **kwargs):
        visitor_data = get_visitor_data(*args, **kwargs)
        preds = visitor_data.predictions[self.path]
        if self.activation is not None:
            preds = self.activation(torch.tensor(preds).float()).detach().cpu().numpy()
        targets = visitor_data.targets[self.path]

        precondition = visitor_data.predictions.get(f'precondition|{self.path}', None)
        if self.available is not None:
            available = self.available(torch.tensor(targets).float()).detach().cpu().numpy()
            if precondition is None:
                precondition = available
            else:
                precondition &= available

        if precondition is None:
            return self.full_result(preds, targets)
        if precondition.sum() == 0:
            return self.empty_result()
        return self.preconditioned_result(preds[precondition], targets[precondition])

    def full_result(self, preds, targets):
        raise NotImplementedError()

    def empty_result(self):
        raise NotImplementedError()

    def preconditioned_result(self, preds, targets):
        raise NotImplementedError()


class CompositeVisitor:

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix=''):
        self.flow = task_flow.get_flow_func()
        self.visitor_out_cls = visitor_out_cls

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = leaf_visitor_cls(task, prefix)
            else:
                instance = CompositeVisitor(task, leaf_visitor_cls, visitor_out_cls, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, data):
        flow_result = self.flow(self, data, self.visitor_out_cls())
        return flow_result


class RootCompositeVisitor:

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix=''):
        self.prefix = prefix
        self.task_flow = task_flow
        self.composite_visitor = CompositeVisitor(task_flow, leaf_visitor_cls, visitor_out_cls, prefix)

    def __call__(self, predictions, targets):
        flow_result = self.composite_visitor(VisitorData(predictions, targets))
        return flow_result.reduce()
