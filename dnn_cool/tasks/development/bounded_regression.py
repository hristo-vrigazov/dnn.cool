import torch

from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.losses.torch import ReducedPerSample
from dnn_cool.metrics import get_default_bounded_regression_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.modules.torch import SigmoidAndMSELoss
from dnn_cool.tasks.development.base import TaskForDevelopment


class BoundedRegressionTaskForDevelopment(TaskForDevelopment):
    """
    Represents a regression task, where the labels are normalized between 0 and 1. Examples include bounding box top
    left corners regression. Here are the defaults:
    * activation - `nn.Sigmoid()` - so that the output is in `[0, 1]`
    * loss - `SigmoidAndMSELoss` - sigmoid on the logits, then standard mean squared error loss.
    """

    def __init__(self, task, labels):
        super().__init__(task,
                         labels,
                         criterion=SigmoidAndMSELoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(SigmoidAndMSELoss(reduction='none')),
                         available_func=positive_values,
                         metrics=get_default_bounded_regression_metrics(),
                         autograd=TorchAutoGrad())
