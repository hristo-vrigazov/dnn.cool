import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Callable, Tuple, Sequence, List

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from treelib import Tree, Node

from dnn_cool.activations import CompositeActivation
from dnn_cool.datasets import FlowDataset, LeafTaskDataset
from dnn_cool.decoders import BinaryDecoder, TaskFlowDecoder, Decoder, ClassificationDecoder, \
    MultilabelClassificationDecoder, NoOpDecoder, BoundedRegressionDecoder
from dnn_cool.evaluation import EvaluationCompositeVisitor, EvaluationVisitor
from dnn_cool.filter import FilterCompositeVisitor, FilterVisitor
from dnn_cool.losses import TaskFlowLoss, ReducedPerSample, TaskFlowLossPerSample
from dnn_cool.metrics import TorchMetric, get_default_binary_metrics, \
    get_default_bounded_regression_metrics, get_default_classification_metrics, \
    get_default_multilabel_classification_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.modules import SigmoidAndMSELoss, Identity, TaskFlowModule, IModuleOutput
from dnn_cool.treelib import TreeExplainer, default_leaf_tree_explainer


class ITask:

    def __init__(self, name: str, inputs, available_func=None, metrics: Tuple[str, TorchMetric] = ()):
        self.name = name
        self.inputs = inputs
        self.available_func = available_func
        self.metrics = metrics

    def get_name(self) -> str:
        return self.name

    def get_activation(self) -> Optional[nn.Module]:
        return None

    def get_decoder(self) -> Decoder:
        return NoOpDecoder()

    def get_filter(self) -> FilterVisitor:
        return FilterVisitor(self, prefix='')

    def get_evaluator(self) -> EvaluationVisitor:
        return EvaluationVisitor(self, prefix='')

    def has_children(self) -> bool:
        return False

    def is_train_only(self) -> bool:
        return False

    def get_available_func(self):
        return self.available_func

    def get_loss(self):
        raise NotImplementedError()

    def get_per_sample_loss(self, prefix='', ctx=None):
        raise NotImplementedError()

    def torch(self):
        raise NotImplementedError()

    def get_inputs(self):
        return self.inputs

    def get_labels(self):
        raise NotImplementedError()

    def get_dataset(self):
        raise NotImplementedError()

    def get_metrics(self):
        for i in range(len(self.metrics)):
            metric_name, metric = self.metrics[i]
            metric.bind_to_task(self)
        return self.metrics

    def get_dropout_mc(self):
        return None

    def __repr__(self):
        params = (
            ('name', self.get_name()),
            ('module', self.torch()),
            ('loss', self.get_loss()),
            ('activation', self.get_activation()),
            ('decoder', self.get_decoder()),
            ('available_func', self.get_available_func()),
            ('per_sample_loss', self.get_per_sample_loss())
        )
        params_str = os.linesep.join(map(lambda x: f'\t{x[0]}={x[1]}', params))
        params_str = f'{os.linesep}{params_str}{os.linesep}'
        res = f'{self.__class__.__module__}.{self.__class__.__name__}({params_str}) at {hex(id(self))}'
        return res


@dataclass
class Task(ITask):
    name: str
    labels: Sequence
    loss: nn.Module
    per_sample_loss: nn.Module
    available_func: Callable
    inputs: Sequence
    activation: Optional[nn.Module]
    decoder: Decoder
    module: nn.Module
    metrics: Sequence[Tuple[str, TorchMetric]]

    def get_activation(self) -> Optional[nn.Module]:
        return self.activation

    def get_decoder(self):
        if self.decoder is None:
            return NoOpDecoder()
        return self.decoder

    def get_loss(self):
        return self.loss

    def get_per_sample_loss(self, prefix='', ctx=None):
        return self.per_sample_loss

    def torch(self):
        return self.module

    def get_inputs(self, *args, **kwargs):
        return self.inputs

    def get_labels(self, *args, **kwargs):
        return self.labels

    def get_dataset(self, **kwargs):
        return LeafTaskDataset(self.inputs, self.labels)

    def get_treelib_explainer(self) -> Callable:
        return default_leaf_tree_explainer


@dataclass()
class BinaryHardcodedTask(Task):
    name: str
    labels: Sequence
    loss: nn.Module = None
    per_sample_loss: nn.Module = None
    available_func: Callable = positive_values
    inputs: Sequence = None
    activation: Optional[nn.Module] = None
    decoder: Decoder = None
    module: nn.Module = Identity()
    metrics: Sequence[Tuple[str, TorchMetric]] = ()


@dataclass()
class BoundedRegressionTask(Task):
    """
    Represents a regression task, where the labels are normalized between 0 and 1. Examples include bounding box top
    left corners regression. Here are the defaults:
    * activation - `nn.Sigmoid()` - so that the output is in `[0, 1]`
    * loss - `SigmoidAndMSELoss` - sigmoid on the logits, then standard mean squared error loss.
    """
    name: str
    labels: Sequence
    loss: nn.Module = field(default_factory=lambda: SigmoidAndMSELoss(reduction='mean'))
    per_sample_loss: nn.Module = field(default_factory=lambda: ReducedPerSample(SigmoidAndMSELoss(reduction='none'),
                                                                                reduction=torch.sum))
    available_func: Callable = field(default_factory=lambda: positive_values)
    module: nn.Module = field(default_factory=lambda: nn.Identity())
    activation: nn.Module = field(default_factory=lambda: nn.Sigmoid())
    decoder: Decoder = field(default_factory=BoundedRegressionDecoder)
    inputs: Sequence = None
    metrics: Sequence[Tuple[str, TorchMetric]] = field(default_factory=get_default_bounded_regression_metrics)


@dataclass()
class BinaryClassificationTask(Task):
    """
    Represents a normal binary classification task. Labels should be between 0 and 1.
    * activation - `nn.Sigmoid()`
    * loss - ``nn.BCEWithLogitsLoss()`
    """
    name: str
    labels: Sequence
    loss: nn.Module = nn.BCEWithLogitsLoss(reduction='mean')
    per_sample_loss: nn.Module = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), reduction=torch.mean)
    available_func: Callable = positive_values
    inputs: Sequence = None
    activation: Optional[nn.Module] = nn.Sigmoid()
    decoder: Decoder = field(default_factory=BinaryDecoder)
    module: nn.Module = Identity()
    metrics: Sequence[Tuple[str, TorchMetric]] = field(default_factory=get_default_binary_metrics)


@dataclass
class ClassificationTask(Task):
    """
    Represents a classification task. Labels should be integers from 0 to N-1, where N is the number of classes
    * activation - `nn.Softmax()`
    * loss - `nn.CrossEntropyLoss()`
    """
    name: str
    labels: Sequence
    loss: nn.Module = nn.CrossEntropyLoss(reduction='mean')
    per_sample_loss: nn.Module = ReducedPerSample(nn.CrossEntropyLoss(reduction='none'), torch.mean)
    available_func: Callable = positive_values
    inputs: Sequence = None
    activation: Optional[nn.Module] = nn.Softmax()
    decoder: Decoder = field(default_factory=ClassificationDecoder)
    module: nn.Module = Identity()
    metrics: Sequence[Tuple[str, TorchMetric]] = field(default_factory=get_default_classification_metrics)

    class_names: Optional = None
    top_k: Optional[int] = 5

    def get_treelib_explainer(self) -> Callable:
        def classification_explainer(task_name: str,
                                     decoded: torch.Tensor,
                                     activated: torch.Tensor,
                                     logits: torch.Tensor,
                                     node_identifier: str) -> Tuple[Tree, Node]:
            tree = Tree()
            start_node = tree.create_node(task_name, node_identifier)
            for i, idx in enumerate(decoded[:self.top_k]):
                name = idx if self.class_names is None else self.class_names[idx]
                description = f'{i}: {name} | activated: {activated[idx]:.4f}, logits: {logits[idx]:.4f}'
                tree.create_node(description, f'{node_identifier}.{idx}', parent=start_node)
            return tree, start_node

        return classification_explainer


@dataclass()
class MultilabelClassificationTask(Task):
    """
    Represents a classification task. Labels should be integers from 0 to N-1, where N is the number of classes
    * activation - `nn.Sigmoid()`
    * loss - `nn.CrossEntropyLoss()`
    """
    name: str
    labels: Sequence
    loss: nn.Module = nn.BCEWithLogitsLoss(reduction='mean')
    per_sample_loss: nn.Module = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), torch.mean)
    available_func: Callable = positive_values
    inputs: Sequence = None
    activation: Optional[nn.Module] = nn.Sigmoid()
    decoder: Decoder = field(default_factory=MultilabelClassificationDecoder)
    module: nn.Module = Identity()
    metrics: Sequence[Tuple[str, TorchMetric]] = field(default_factory=get_default_multilabel_classification_metrics)

    class_names: Optional = None

    def get_treelib_explainer(self) -> Callable:
        def explainer(task_name: str,
                      decoded: np.ndarray,
                      activated: np.ndarray,
                      logits: np.ndarray,
                      node_identifier: str) -> Tuple[Tree, Node]:
            tree = Tree()
            start_node = tree.create_node(task_name, node_identifier)
            for i, val in enumerate(decoded):
                name = i if self.class_names is None else self.class_names[i]
                description = f'{i}: {name} | decoded: {decoded[i]}, ' \
                              f'activated: {activated[i]:.4f}, ' \
                              f'logits: {logits[i]:.4f}'
                tree.create_node(description, f'{node_identifier}.{i}', parent=start_node)
            return tree, start_node

        return explainer


class TaskFlow(ITask):

    def __init__(self, name, tasks: Iterable[ITask], flow_func=None, inputs=None, available_func=None):
        super().__init__(name, inputs, available_func)
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task
        if flow_func is not None:
            self._flow_func = flow_func
        self.ctx = {}

    def get_loss(self):
        return TaskFlowLoss(self, ctx=self.ctx)

    def get_per_sample_loss(self, prefix='', ctx=None):
        if ctx is None:
            ctx = self.ctx
        return TaskFlowLossPerSample(self, prefix=prefix, ctx=ctx)

    def torch(self):
        return TaskFlowModule(self)

    def has_children(self):
        return True

    def get_dataset(self, **kwargs) -> Dataset:
        return FlowDataset(self, **kwargs)

    def flow(self, x, out):
        raise NotImplementedError()

    def get_flow_func(self):
        if hasattr(self, '_flow_func') and self._flow_func is not None:
            return self._flow_func
        return self.__class__.flow

    def get_metrics(self):
        all_metrics = []
        for task in self.tasks.values():
            all_metrics += task.get_metrics()
        return all_metrics

    def get_treelib_explainer(self):
        return TreeExplainer(self.get_name(), self.get_flow_func(), self.tasks)

    def get_labels(self, *args, **kwargs):
        all_labels = []
        for task in self.tasks.values():
            all_labels += task.get_labels()
        return all_labels

    def get_decoder(self):
        return TaskFlowDecoder(self)

    def get_activation(self) -> Callable:
        return CompositeActivation(self)

    def get_evaluator(self):
        return EvaluationCompositeVisitor(self, prefix='')

    def get_filter(self):
        return FilterCompositeVisitor(self, prefix='')

    def get_all_children(self, prefix=''):
        tasks = {}
        for task_name, task in self.tasks.items():
            if task.has_children():
                assert isinstance(task, TaskFlow)
                tasks.update(task.get_all_children(prefix=f'{prefix}{task.get_name()}.'))
            else:
                tasks[prefix + task_name] = task
        return tasks
