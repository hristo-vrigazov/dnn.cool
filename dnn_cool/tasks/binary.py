from typing import Callable

import torch
from torch import nn

from dnn_cool.decoders.base import NoOpDecoder
from dnn_cool.decoders.binary import BinaryDecoder
from dnn_cool.tasks.base import Task


class BinaryClassificationTask(Task):

    def __init__(self, name, torch_module, dropout_mc=None):
        super().__init__(name,
                         torch_module=torch_module,
                         activation=nn.Sigmoid(),
                         decoder=BinaryDecoder(),
                         dropout_mc=dropout_mc)


class BinaryHardcodedTask(Task):

    def __init__(self, name):
        super().__init__(name, torch_module=None, activation=None, decoder=NoOpDecoder(), dropout_mc=None)

