import torch
from torch import nn

from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.losses.torch import ReducedPerSample
from dnn_cool.metrics import get_default_classification_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.tasks.development.base import TaskForDevelopment


class ClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, task, labels):
        super().__init__(task,
                         labels,
                         criterion=nn.CrossEntropyLoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(nn.CrossEntropyLoss(reduction='none')),
                         available_func=positive_values,
                         metrics=get_default_classification_metrics(),
                         autograd=TorchAutoGrad())
