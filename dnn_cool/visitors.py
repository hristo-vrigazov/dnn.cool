from dataclasses import dataclass
from typing import Dict

import numpy as np

from dnn_cool.dsl import IFeaturesDict, IOut, IFlowTaskResult, ICondition, IFlowTask
from dnn_cool.external.autograd import IAutoGrad, squeeze_last_dim_if_needed


def get_visitor_data(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, VisitorData):
            return arg


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


class LeafVisitor(IFlowTask):

    def __init__(self, task, prefix, autograd: IAutoGrad):
        self.activation = task.get_minimal().get_activation()
        self.decoder = task.get_minimal().get_decoder()
        self.task_is_train_only = task.get_minimal().is_train_only()
        self.prefix = prefix
        self.path = self.prefix + task.get_name()
        self.autograd = autograd

    def __call__(self, *args, **kwargs):
        visitor_data = get_visitor_data(*args, **kwargs)
        preds = visitor_data.predictions.get(self.path)
        if preds is None:
            if not self.task_is_train_only:
                raise ValueError(f'The task {self.path} has no predictions, but it is not marked as train only.')
            return self.empty_result()
        if self.activation is not None:
            preds = self.activation(self.autograd.as_float(preds))
            preds = self.autograd.to_numpy(preds)
        targets = visitor_data.targets[self.path] if visitor_data.targets is not None else None

        precondition = self.get_precondition(visitor_data, preds)
        if precondition.sum() == 0:
            return self.empty_result()
        precondition = squeeze_last_dim_if_needed(precondition)
        preconditioned_targets = targets[precondition] if targets is not None else None
        return self.preconditioned_result(preds[precondition], preconditioned_targets)

    def get_precondition(self, visitor_data, preds):
        precondition = visitor_data.predictions.get(f'precondition|{self.path}')
        if precondition is not None:
            return precondition
        return np.ones(preds.shape[:-1], dtype=bool)

    def empty_result(self):
        raise NotImplementedError()

    def preconditioned_result(self, preds, targets):
        raise NotImplementedError()


class CompositeVisitor(IFlowTask):

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix: str, autograd: IAutoGrad):
        self.flow = task_flow.get_flow_func()
        self.visitor_out_cls = visitor_out_cls

        for key, task in task_flow.tasks.items():
            if not task.get_minimal().has_children():
                instance = leaf_visitor_cls(task, prefix, autograd)
            else:
                instance = CompositeVisitor(task, leaf_visitor_cls, visitor_out_cls,
                                            prefix=f'{prefix}{task.get_name()}.',
                                            autograd=autograd)
            setattr(self, key, instance)

    def __call__(self, data) -> VisitorOut:
        flow_result = self.flow(self, data, self.visitor_out_cls())
        return flow_result


class RootCompositeVisitor(IFlowTask):

    def __init__(self, task_flow, leaf_visitor_cls, visitor_out_cls, prefix, autograd):
        self.prefix = prefix
        self.task_flow = task_flow
        self.composite_visitor = CompositeVisitor(task_flow=task_flow,
                                                  leaf_visitor_cls=leaf_visitor_cls,
                                                  visitor_out_cls=visitor_out_cls,
                                                  prefix=prefix,
                                                  autograd=autograd)

    def __call__(self, predictions, targets=None):
        flow_result = self.composite_visitor(VisitorData(predictions, targets))
        return flow_result.reduce()
