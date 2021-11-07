from torch import nn

from dnn_cool.decoders.bounded_regression import BoundedRegressionDecoder
from dnn_cool.tasks.base import Task


class BoundedRegressionTask(Task):

    def __init__(self, name, torch_module, scale, dropout_mc=None):
        super().__init__(name, torch_module=torch_module, activation=nn.Sigmoid(),
                         decoder=BoundedRegressionDecoder(scale=scale),
                         dropout_mc=dropout_mc)