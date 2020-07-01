import torch
from torch import nn


class LossFlowData:

    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    # To be compatible with the pipeline
    def __getattr__(self, item):
        return self


class LossItems:

    def __init__(self, loss_items):
        self.loss_items = loss_items

    def __add__(self, other):
        return LossItems(self.loss_items + other.loss_items)

    # None of the methods below modify the state. They are here
    # to be compatible with the pipeline
    def __getattr__(self, item):
        return self

    def __or__(self, result):
        return self

    def __invert__(self):
        return self

    def __and__(self, other):
        return self


def get_flow_data(*args, **kwargs):
    for arg in args:
        if isinstance(arg, LossFlowData):
            return arg

    for arg in kwargs.values():
        if isinstance(arg, LossFlowData):
            return arg


def squeeze_if_needed(tensor):
    if len(tensor.shape) > 2:
        raise ValueError(f'Trying to squeeze the second dimension out of a tensor with shape: {tensor.shape}')
    if len(tensor.shape) == 2:
        return tensor.squeeze(dim=1)
    return tensor


class TaskLossDecorator(nn.Module):

    def __init__(self, task, child_reduction):
        super().__init__()
        self.task_name = task.get_name()

        kwargs = {}
        if task.has_children():
            kwargs['parent_reduction'] = child_reduction
            kwargs['child_reduction'] = child_reduction
        else:
            kwargs['reduction'] = child_reduction
        self.loss = task.loss(**kwargs)

    def forward(self, *args, **kwargs):
        return LossItems(self.read_vectors(args, kwargs))

    def read_vectors(self, args, kwargs):
        loss_flow_data = get_flow_data(*args, **kwargs)
        outputs = loss_flow_data.outputs[self.task_name]
        precondition = loss_flow_data.outputs.get(f'precondition|{self.task_name}', None)
        targets = loss_flow_data.targets[self.task_name]
        n = len(outputs)
        loss_items = torch.zeros(n, dtype=outputs.dtype, device=outputs.device)
        if precondition is None:
            return squeeze_if_needed(self.loss(outputs, targets))
        if precondition.sum() == 0:
            return squeeze_if_needed(loss_items)
        precondition = squeeze_if_needed(precondition)
        loss_items[precondition] = squeeze_if_needed(self.loss(outputs[precondition], targets[precondition]))
        return loss_items


class TaskFlowLoss(nn.Module):

    def __init__(self, task_flow, parent_reduction, child_reduction):
        super().__init__()
        self.parent_reduction = parent_reduction
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.__class__.flow

        for key, task in task_flow.tasks.items():
            setattr(self, key, TaskLossDecorator(task, child_reduction))

        print('Party')

    def forward(self, outputs, targets):
        value = self.__any_value(outputs)
        n = len(value)
        loss_items = torch.zeros(n, dtype=value.dtype, device=value.device)
        flow_result = self.flow(self, LossFlowData(outputs, targets), LossItems(loss_items))

        if self.parent_reduction == 'mean':
            return flow_result.loss_items.mean()

        return flow_result.loss_items

    def __any_value(self, outputs):
        for key, value in outputs.items():
            if not key.startswith('precondition'):
                return value
