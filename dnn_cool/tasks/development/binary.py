import torch
from torch import nn

from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.losses.torch import ReducedPerSample
from dnn_cool.metrics import get_default_binary_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.tasks.development.base import TaskForDevelopment


class BinaryHardcodedTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels=labels,
                         criterion=None,
                         per_sample_criterion=None,
                         available_func=positive_values,
                         metrics=[],
                         autograd=TorchAutoGrad())


class BinaryClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        reduced_per_sample = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), reduction=torch.mean)
        super().__init__(name,
                         labels,
                         criterion=nn.BCEWithLogitsLoss(reduction='mean'),
                         per_sample_criterion=reduced_per_sample,
                         available_func=positive_values,
                         metrics=get_default_binary_metrics(),
                         autograd=TorchAutoGrad())
