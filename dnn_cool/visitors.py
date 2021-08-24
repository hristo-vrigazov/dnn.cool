import torch

from dataclasses import dataclass
from typing import Dict

from dnn_cool.dsl import ICondition, IOut, IFlowTaskResult, IFlowTask, IFeaturesDict
from dnn_cool.losses import squeeze_if_needed


@dataclass
class VisitorData(IFeaturesDict):
    predictions: Dict
    targets: Dict

    def __getattr__(self, item):
        return self


class VisitorOut(IOut, IFlowTaskResult, ICondition):

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

    # Pipeline compatibility
    def __and__(self, other):
        return self

    def reduce(self):
        raise NotImplementedError()


def get_visitor_data(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, VisitorData):
            return arg


class LeafVisitor(IFlowTask):

    def __init__(self, task, prefix):
        self.activation = task.get_minimal().get_activation()
        self.decoder = task.get_minimal().get_decoder()
        self.task_is_train_only = task.get_minimal().is_train_only()
        self.prefix = prefix
        self.path = self.prefix + task.get_name()

    def __call__(self, *args, **kwargs):
        visitor_data = get_visitor_data(*args, **kwargs)
        preds = visitor_data.predictions.get(self.path)
        if preds is None:
            if not self.task_is_train_only:
                raise ValueError(f'The task {self.path} has no predictions, but it is not marked as train only.')
            return self.empty_result()
        if self.activation is not None:
            preds = self.activation(torch.tensor(preds).float()).detach().cpu().numpy()
        targets = visitor_data.targets[self.path] if visitor_data.targets is not None else None

        precondition = visitor_data.predictions[f'precondition|{self.path}']
        if precondition.sum() == 0:
            return self.empty_result()
        precondition = squeeze_if_needed(precondition)
        preconditioned_targets = targets[precondition] if targets is not None else None
        return self.preconditioned_result(preds[precondition], preconditioned_targets)

    def full_result(self, preds, targets):
        raise NotImplementedError()

    def empty_result(self):
        raise NotImplementedError()

    def preconditioned_result(self, preds, targets):
        raise NotImplementedError()


class CompositeVisitor(IFlowTask):

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix=''):
        self.flow = task_flow.get_flow_func()
        self.visitor_out_cls = visitor_out_cls

        for key, task in task_flow.tasks.items():
            if not task.get_minimal().has_children():
                instance = leaf_visitor_cls(task, prefix)
            else:
                instance = CompositeVisitor(task, leaf_visitor_cls, visitor_out_cls,
                                            prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, data) -> VisitorOut:
        flow_result = self.flow(self, data, self.visitor_out_cls())
        return flow_result


class RootCompositeVisitor(IFlowTask):

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix=''):
        self.prefix = prefix
        self.task_flow = task_flow
        self.composite_visitor = CompositeVisitor(task_flow, leaf_visitor_cls, visitor_out_cls, prefix)

    def __call__(self, predictions, targets=None):
        flow_result = self.composite_visitor(VisitorData(predictions, targets))
        return flow_result.reduce()
