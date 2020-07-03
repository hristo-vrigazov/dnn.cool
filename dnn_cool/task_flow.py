from abc import ABC
from typing import Iterable, Dict, Optional

from torch import nn
from torch.utils.data import Dataset

from dnn_cool.datasets import FlowDataset
from dnn_cool.losses import TaskFlowLoss
from dnn_cool.modules import SigmoidAndMSELoss, Identity, NestedFC, TaskFlowModule
from functools import partial


class Result:

    def __init__(self, task, *args, **kwargs):
        self.precondition = None
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def __or__(self, result):
        self.precondition = result
        return self

    def torch(self) -> nn.Module:
        """
        This method returns a new instance of a Pytorch module, which takes into account the precondition of the task.
        In train mode, the precondition is evaluated based on the ground truth.
        In eval mode, the precondition is evaluated based on the predictions for the precondition task.
        :return:
        """
        return self.task.torch()


class BooleanResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)

    def __invert__(self):
        return self

    def __and__(self, other):
        return self


class LocalizationResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class ClassificationResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class NestedClassificationResult(Result):
    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class RegressionResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class NestedResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        self.res = {}
        key = kwargs.get('key', None)
        if key is not None:
            self.res[key] = kwargs.get('value', None)

    def __iadd__(self, other):
        self.res.update(other.res)
        return self

    def __getattr__(self, attr):
        return self.res[attr]

    def __or__(self, result):
        super().__or__(result)
        for key, value in self.res.items():
            self.res[key] = value | result
        return self

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class Task:

    def __init__(self, name: str, module_options: Optional[Dict] = None):
        self.name = name
        self.module_options = module_options if module_options is not None else {}

    def __call__(self, *args, **kwargs) -> Result:
        return NestedResult(task=self, key=self.name, value=self.do_call(*args, **kwargs))

    def do_call(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        return self.name

    def activation(self) -> Optional[nn.Module]:
        pass

    def decoder(self):
        pass

    def encoder(self):
        pass

    def has_children(self):
        return False

    def loss(self, *args, **kwargs):
        raise NotImplementedError()

    def torch(self):
        """
        Creates a new instance of a Pytorch module, that has to be invoked for those parts of the input, for which
        the precondition is satisfied.
        :return:
        """
        raise NotImplementedError()

    def get_inputs(self, *args, **kwargs):
        raise NotImplementedError()

    def get_labels(self, **kwargs):
        raise NotImplementedError()

    def metrics(self):
        return []


class BinaryHardcodedTask(Task):

    def do_call(self, *args, **kwargs) -> BooleanResult:
        return BooleanResult(self, *args, **kwargs)

    def torch(self):
        return Identity()

    def activation(self) -> Optional[nn.Module]:
        return None


class LocalizationTask(Task):
    def do_call(self, *args, **kwargs):
        return LocalizationResult(self, *args, **kwargs)

    def torch(self) -> nn.Module:
        return nn.Linear(self.module_options['in_features'], 4, self.module_options.get('bias', True))

    def activation(self) -> nn.Module:
        return nn.Sigmoid()

    def loss(self, *args, **kwargs):
        return SigmoidAndMSELoss(*args, **kwargs)


class BinaryClassificationTask(Task):

    def do_call(self, *args, **kwargs) -> BooleanResult:
        return BooleanResult(self, *args, **kwargs)

    def torch(self) -> nn.Module:
        return nn.Linear(self.module_options['in_features'], 1, self.module_options.get('bias', True))

    def activation(self) -> nn.Module:
        return nn.Sigmoid()

    def decoder(self):
        return self.decode

    def decode(self, x):
        return x > 0.5

    def loss(self, *args, **kwargs):
        return nn.BCEWithLogitsLoss(*args, **kwargs)

    def metrics(self):
        from catalyst.utils.metrics import accuracy

        def single_result_accuracy(outputs, targets, threshold=0.5, activation='Sigmoid'):
            return accuracy(outputs, targets, threshold=threshold, activation=activation)[0]

        return [
            ('acc_0.5', partial(single_result_accuracy, threshold=0.5, activation='Sigmoid'))
        ]


class ClassificationTask(Task):

    def do_call(self, *args, **kwargs):
        return ClassificationResult(self, *args, **kwargs)

    def torch(self) -> nn.Module:
        return nn.Linear(self.module_options['in_features'],
                         self.module_options['out_features'],
                         self.module_options.get('bias', True))

    def activation(self) -> nn.Module:
        return nn.Softmax(dim=-1)

    def decoder(self):
        return self.decode

    def decode(self, x):
        return (-x).argsort(dim=1)

    def loss(self, *args, **kwargs):
        return nn.CrossEntropyLoss(*args, **kwargs)

    def metrics(self):
        from catalyst.utils.metrics import accuracy

        def single_result_accuracy(outputs, targets, **kwargs):
            return accuracy(outputs, targets, **kwargs)[0]

        return [
            ('top_1_acc', partial(single_result_accuracy, topk=(1,), activation='Softmax')),
            ('top_3_acc', partial(single_result_accuracy, topk=(3,), activation='Softmax')),
        ]


class RegressionTask(Task):

    def __init__(self, name: str, module_options, activation_func):
        super().__init__(name, module_options)
        self.activation_func = activation_func

    def do_call(self, *args, **kwargs):
        return RegressionResult(self, *args, **kwargs)

    def torch(self):
        return nn.Linear(self.module_options['in_features'],
                         self.module_options.get('out_features', 1),
                         self.module_options.get('bias', True))

    def activation(self) -> nn.Module:
        return self.activation_func

    def loss(self, *args, **kwargs):
        if isinstance(self.activation_func, nn.Sigmoid):
            return SigmoidAndMSELoss(*args, **kwargs)
        return nn.MSELoss(*args, **kwargs)


class NestedClassificationTask(Task):

    def __init__(self, name, top_k, module_options):
        super().__init__(name, module_options)
        self.top_k = top_k

    def do_call(self, *args, **kwargs):
        return NestedClassificationResult(self, *args, **kwargs)

    def loss(self, *args, **kwargs):
        pass

    def torch(self):
        return NestedFC(self.module_options['in_features'],
                        self.module_options['out_features_nested'],
                        self.module_options['bias'],
                        self.top_k)


class TaskFlow(Task):

    def __init__(self, name, tasks: Iterable[Task]):
        super().__init__(name)
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task

    def __getattr__(self, attr):
        return self.tasks[attr]

    def trace_flow(self, x):
        out = NestedResult(self)
        return self.flow(x, out)

    def do_call(self, *args, **kwargs):
        return NestedResult(task=self)

    def activation(self) -> Optional[nn.Module]:
        pass

    def loss(self, **kwargs):
        return TaskFlowLoss(self, **kwargs)

    def torch(self):
        return TaskFlowModule(self)

    def has_children(self):
        return True

    def get_labels(self, **kwargs) -> Dataset:
        return FlowDataset(self, **kwargs)

    def flow(self, x, out):
        raise NotImplementedError()

    def metrics(self):
        all_metrics = []
        for task in self.tasks.values():
            all_metrics += task.metrics()
        return all_metrics
