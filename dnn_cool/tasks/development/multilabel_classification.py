import torch
from torch import nn

from dnn_cool.losses import ReducedPerSample
from dnn_cool.metrics import get_default_multilabel_classification_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.tasks.development.base import TaskForDevelopment


class MultilabelClassificationTaskForDevelopment(TaskForDevelopment):
    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels,
                         criterion=nn.BCEWithLogitsLoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), torch.mean),
                         available_func=positive_values,
                         metrics=get_default_multilabel_classification_metrics())
