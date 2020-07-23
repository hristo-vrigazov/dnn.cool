from abc import ABC
from typing import Iterable, Dict, Optional, Callable, List, Tuple

from torch import nn
from torch.utils.data import Dataset

from dnn_cool.datasets import FlowDataset
from dnn_cool.decoders import threshold_binary, sort_declining
from dnn_cool.losses import TaskFlowLoss
from dnn_cool.metrics import single_result_accuracy
from dnn_cool.modules import SigmoidAndMSELoss, Identity, NestedFC, TaskFlowModule
from functools import partial


class Values:
    pass


class ITask:

    def __init__(self, name: str, inputs: Values):
        self.name = name
        self.__inputs = inputs

    def get_name(self):
        return self.name

    def get_activation(self) -> Optional[nn.Module]:
        return None

    def get_decoder(self):
        return None

    def has_children(self):
        return False

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def torch(self):
        """
        Creates a new instance of a Pytorch module, that has to be invoked for those parts of the input, for which
        the precondition is satisfied.
        :return:
        """
        raise NotImplementedError()

    def get_inputs(self, *args, **kwargs):
        return self.__inputs

    def get_labels(self, **kwargs):
        raise NotImplementedError()

    def get_metrics(self):
        return []


class Task(ITask):

    def __init__(self,
                 name: str,
                 activation: Optional[nn.Module],
                 decoder: Callable,
                 loss: Callable,
                 module: nn.Module,
                 inputs: Values,
                 labels: Values,
                 metrics: List[Tuple[str, Callable]]):
        super().__init__(name, inputs)
        self._activation = activation
        self._decoder = decoder
        self._loss = loss
        self._module = module
        self._inputs = inputs
        self._labels = labels
        self._metrics = metrics

    def get_activation(self) -> Optional[nn.Module]:
        return self._activation

    def get_decoder(self):
        return self._decoder

    def get_loss(self, *args, **kwargs):
        return self._loss(*args, **kwargs)

    def torch(self):
        return self._module

    def get_inputs(self, *args, **kwargs):
        return self._inputs

    def get_labels(self, **kwargs):
        return self._labels

    def get_metrics(self):
        return self._metrics


class BinaryHardcodedTask(Task):

    def __init__(self,
                 name: str,
                 labels: Values,
                 activation: Optional[nn.Module] = None,
                 decoder: Callable = None,
                 loss: Callable = None,
                 module: nn.Module = Identity(),
                 inputs: Values = None,
                 metrics=()):
        super().__init__(name, activation, decoder, loss, module, inputs, labels, metrics)


class BoundedRegressionTask(Task):
    """
    Represents a regression task, where the labels are normalized between 0 and 1. Examples include bounding box top
    left corners regression. Here are the defaults:
    * activation - `nn.Sigmoid()` - so that the output is in `[0, 1]`
    * loss - `SigmoidAndMSELoss` - sigmoid on the logits, then standard mean squared error loss.
    """

    def __init__(self,
                 name: str,
                 module: nn.Module,
                 labels: Values,
                 activation: Optional[nn.Module] = nn.Sigmoid(),
                 decoder: Callable = None,
                 loss=SigmoidAndMSELoss,
                 inputs=None,
                 metrics=()):
        super().__init__(name, activation, decoder, loss, module, inputs, labels, metrics)


class BinaryClassificationTask(Task):
    """
    Represents a normal binary classification task. Labels should be between 0 and 1.
    * activation - `nn.Sigmoid()`
    * loss - ``nn.BCEWithLogitsLoss()`
    """

    def __init__(self,
                 name: str,
                 module: nn.Module,
                 labels: Values,
                 activation: Optional[nn.Module] = nn.Sigmoid(),
                 decoder: Callable = threshold_binary,
                 loss=nn.BCEWithLogitsLoss,
                 inputs: Values = None,
                 metrics=(
                         ('acc_0.5', partial(single_result_accuracy, threshold=0.5, activation='Sigmoid')),
                 )):
        super().__init__(name, activation, decoder, loss, module, inputs, labels, metrics)


class ClassificationTask(Task):
    """
    Represents a classification task. Labels should be integers from 0 to N-1, where N is the number of classes
    * activation - `nn.Softmax(dim=-1)`
    * loss - `nn.CrossEntropyLoss()`
    """

    def __init__(self,
                 name: str,
                 module: nn.Module,
                 inputs: Values,
                 labels: Values,
                 activation=nn.Softmax(dim=-1),
                 decoder=sort_declining,
                 loss=nn.CrossEntropyLoss,
                 metrics=(
                         ('top_1_acc', partial(single_result_accuracy, topk=(1,), activation='Softmax')),
                         ('top_3_acc', partial(single_result_accuracy, topk=(3,), activation='Softmax')),
                 )):
        super().__init__(name, activation, decoder, loss, module, inputs, labels, metrics)


class RegressionTask(ITask):

    def __init__(self, name: str, inputs: Values, labels, activation_func):
        super().__init__(name, inputs)
        self.activation_func = activation_func
        self.labels = labels

    def torch(self):
        return nn.Linear(128, 1)

    def get_activation(self) -> nn.Module:
        return self.activation_func

    def get_loss(self, *args, **kwargs):
        if isinstance(self.activation_func, nn.Sigmoid):
            return SigmoidAndMSELoss(*args, **kwargs)
        return nn.MSELoss(*args, **kwargs)

    def get_labels(self, **kwargs):
        return self.labels


class NestedClassificationTask(ITask):

    def __init__(self, name, inputs, labels, top_k):
        super().__init__(name, inputs)
        self.top_k = top_k
        self.labels = labels

    def get_loss(self, *args, **kwargs):
        pass

    def torch(self):
        return NestedFC(128, [9, 15, 2, 12], True, self.top_k)

    def get_labels(self, **kwargs):
        return self.labels


class TaskFlow(ITask):

    def __init__(self, name, tasks: Iterable[ITask], inputs=None, flow_func=None):
        super().__init__(name, inputs)
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task
        self.__flow_func = flow_func

    def __getattr__(self, attr):
        return self.tasks[attr]

    def get_activation(self) -> Optional[nn.Module]:
        pass

    def get_loss(self, **kwargs):
        return TaskFlowLoss(self, **kwargs)

    def torch(self):
        return TaskFlowModule(self)

    def has_children(self):
        return True

    def get_labels(self, **kwargs) -> Dataset:
        return FlowDataset(self, **kwargs)

    def flow(self, x, out):
        if self.__flow_func is not None:
            self.__flow_func(x, out)
        raise NotImplementedError()

    def get_metrics(self):
        all_metrics = []
        for task in self.tasks.values():
            all_metrics += task.get_metrics()
        return all_metrics
