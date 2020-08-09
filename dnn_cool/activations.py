from torch import nn


class TaskFlowActivation(nn.Module):

    def __init__(self, task_flow):
        super().__init__()
        self.task_flow = task_flow

    def forward(self):
        raise NotImplementedError()