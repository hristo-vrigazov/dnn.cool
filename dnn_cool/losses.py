import torch
from torch import nn


class LossFlowData:

    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets


class TaskLossDecorator(nn.Module):

    def __init__(self, task, reduction):
        super().__init__()
        self.loss = task.loss(reduction=reduction)

    def forward(self, *args, **kwargs):
        pass


class TaskFlowLoss(nn.Module):

    def __init__(self, task_flow, reduction):
        super().__init__()
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.__class__.flow

        for key, task in task_flow.tasks.items():
            setattr(self, key, TaskLossDecorator(task, reduction))

    def forward(self, outputs, targets):
        value = self.__any_value(outputs)
        n = len(value)
        loss_items = torch.zeros(n, dtype=value.dtype, device=value.device)
        return self.flow(self, LossFlowData(outputs, targets), loss_items)

    def __any_value(self, outputs):
        for key, value in outputs.items():
            if not key.startswith('precondition'):
                return value
