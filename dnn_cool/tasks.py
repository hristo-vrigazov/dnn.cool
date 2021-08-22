from pathlib import Path
from typing import Iterable, Optional, Callable, Tuple, List, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from treelib import Tree

from dnn_cool.activations import CompositeActivation
from dnn_cool.datasets import FlowDataset
from dnn_cool.decoders import BinaryDecoder, TaskFlowDecoder, Decoder, ClassificationDecoder, \
    MultilabelClassificationDecoder, NoOpDecoder, BoundedRegressionDecoder
from dnn_cool.evaluation import EvaluationCompositeVisitor, EvaluationVisitor
from dnn_cool.filter import FilterCompositeVisitor, FilterVisitor
from dnn_cool.losses import TaskFlowCriterion, ReducedPerSample, TaskFlowLossPerSample
from dnn_cool.metrics import TorchMetric, get_default_binary_metrics, \
    get_default_bounded_regression_metrics, get_default_classification_metrics, \
    get_default_multilabel_classification_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.modules import SigmoidAndMSELoss, TaskFlowModule
from dnn_cool.treelib import TreeExplainer, default_leaf_tree_explainer


class MinimalTask:

    def __init__(self, name, torch_module, activation, decoder, dropout_mc, treelib_explainer=None):
        self.name = name
        self.activation = activation
        self.decoder = decoder
        self.torch_module = torch_module
        self.dropout_mc = dropout_mc
        self.treelib_explainer = treelib_explainer

    def get_name(self) -> str:
        return self.name

    def get_activation(self) -> Optional[nn.Module]:
        return self.activation

    def get_decoder(self) -> Decoder:
        return self.decoder

    def has_children(self) -> bool:
        return False

    def is_train_only(self) -> bool:
        return False

    def torch(self):
        return self.torch_module

    def get_dropout_mc(self):
        return self.dropout_mc

    def get_treelib_explainer(self) -> Callable:
        return default_leaf_tree_explainer if self.treelib_explainer is None else self.treelib_explainer


class TaskForDevelopment:

    def __init__(self, name: str,
                 labels,
                 criterion,
                 per_sample_criterion,
                 available_func,
                 metrics: List[Tuple[str, TorchMetric]]):
        self.name = name
        self.labels = labels
        self.criterion = criterion
        self.per_sample_criterion = per_sample_criterion
        self.available_func = available_func
        self.metrics = metrics if metrics is not None else []

    def get_name(self) -> str:
        return self.name

    def get_filter(self) -> FilterVisitor:
        return FilterVisitor(self, prefix='')

    def get_evaluator(self) -> EvaluationVisitor:
        return EvaluationVisitor(self, prefix='')

    def get_available_func(self):
        return self.available_func

    def get_criterion(self):
        return self.criterion

    def get_per_sample_criterion(self):
        return self.per_sample_criterion

    def get_labels(self):
        return self.labels

    def get_metrics(self):
        for i in range(len(self.metrics)):
            metric_name, metric = self.metrics[i]
            metric.bind_to_task(self)
        return self.metrics


class BinaryHardcodedTaskMinimal(MinimalTask):

    def __init__(self, name):
        super().__init__(name, torch_module=None, activation=None, decoder=NoOpDecoder(), dropout_mc=None)


class BinaryHardcodedTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels=labels,
                         criterion=None,
                         per_sample_criterion=None,
                         available_func=positive_values,
                         metrics=[])


class BoundedRegressionTaskMinimal(MinimalTask):

    def __init__(self, name, torch_module, dropout_mc=None):
        super().__init__(name, torch_module=torch_module, activation=nn.Sigmoid(), decoder=BoundedRegressionDecoder(),
                         dropout_mc=dropout_mc)


class BoundedRegressionTaskForDevelopment(TaskForDevelopment):
    """
    Represents a regression task, where the labels are normalized between 0 and 1. Examples include bounding box top
    left corners regression. Here are the defaults:
    * activation - `nn.Sigmoid()` - so that the output is in `[0, 1]`
    * loss - `SigmoidAndMSELoss` - sigmoid on the logits, then standard mean squared error loss.
    """

    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels,
                         criterion=SigmoidAndMSELoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(SigmoidAndMSELoss(reduction='none'),
                                                               reduction=torch.sum),
                         available_func=positive_values,
                         metrics=get_default_bounded_regression_metrics())


class BinaryClassificationTaskMinimal(MinimalTask):

    def __init__(self, name, torch_module, dropout_mc=None):
        super().__init__(name,
                         torch_module=torch_module,
                         activation=nn.Sigmoid(),
                         decoder=BinaryDecoder(),
                         dropout_mc=dropout_mc)


class BinaryClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        reduced_per_sample = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), reduction=torch.mean)
        super().__init__(name,
                         labels,
                         criterion=nn.BCEWithLogitsLoss(reduction='mean'),
                         per_sample_criterion=reduced_per_sample,
                         available_func=positive_values,
                         metrics=get_default_binary_metrics())


class ClassificationTaskMinimal(MinimalTask):

    def __init__(self, name, torch_module,
                 class_names: Optional[List[str]] = None,
                 top_k: Optional[int] = 5,
                 dropout_mc=None):
        super().__init__(name,
                         torch_module=torch_module,
                         activation=nn.Softmax(dim=-1),
                         decoder=ClassificationDecoder,
                         dropout_mc=dropout_mc)
        self.class_names: List[str] = class_names
        self.top_k = top_k

    def get_treelib_explainer(self) -> Callable:
        return self.generate_tree

    def generate_tree(self, task_name: str,
                      decoded: torch.Tensor,
                      activated: torch.Tensor,
                      logits: torch.Tensor,
                      node_identifier: str):
        tree = Tree()
        start_node = tree.create_node(task_name, node_identifier)
        for i, idx in enumerate(decoded[:self.top_k]):
            name = idx if self.class_names is None else self.class_names[idx]
            description = f'{i}: {name} | activated: {activated[idx]:.4f}, logits: {logits[idx]:.4f}'
            tree.create_node(description, f'{node_identifier}.{idx}', parent=start_node)
        return tree, start_node


class ClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str,
                 labels):
        super().__init__(name,
                         labels,
                         criterion=nn.CrossEntropyLoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(nn.CrossEntropyLoss(reduction='none'), torch.mean),
                         available_func=positive_values,
                         metrics=get_default_classification_metrics())


class MultilabelClassificationTaskMinimal(MinimalTask):

    def __init__(self, name, torch_module,
                 class_names: Optional[List[str]] = None,
                 dropout_mc=None):
        super().__init__(name,
                         torch_module,
                         activation=nn.Sigmoid(),
                         decoder=MultilabelClassificationDecoder(),
                         dropout_mc=dropout_mc)
        self.class_names = class_names

    def get_treelib_explainer(self) -> Callable:
        return self.generate_tree

    def generate_tree(self, task_name: str,
                      decoded: torch.Tensor,
                      activated: torch.Tensor,
                      logits: torch.Tensor,
                      node_identifier: str):
        tree = Tree()
        start_node = tree.create_node(task_name, node_identifier)
        for i, val in enumerate(decoded):
            name = i if self.class_names is None else self.class_names[i]
            description = f'{i}: {name} | decoded: {decoded[i]}, ' \
                          f'activated: {activated[i]:.4f}, ' \
                          f'logits: {logits[i]:.4f}'
            tree.create_node(description, f'{node_identifier}.{i}', parent=start_node)
        return tree, start_node


class MultilabelClassificationTaskForDevelopment(TaskForDevelopment):
    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels,
                         criterion=nn.BCEWithLogitsLoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), torch.mean),
                         available_func=positive_values,
                         metrics=get_default_multilabel_classification_metrics())


class TaskFlowBase:

    def __init__(self, name, tasks, flow_func):
        self.name = name
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task
        if flow_func is not None:
            self._flow_func = flow_func
        self.ctx = {}

    def get_name(self):
        return self.name

    def get_flow_func(self):
        return self._flow_func

    def get_all_children(self, prefix=''):
        tasks = {}
        for task_name, task in self.tasks.items():
            if task.has_children():
                assert isinstance(task, TaskFlowBase)
                tasks.update(task.get_all_children(prefix=f'{prefix}{task.get_name()}.'))
            else:
                tasks[prefix + task_name] = task
        return tasks


class TaskFlowMinimal(MinimalTask, TaskFlowBase):

    def __init__(self, name, tasks: Iterable[MinimalTask], flow_func, dropout_mc=None):
        TaskFlowBase.__init__(self, name, tasks, flow_func)
        MinimalTask.__init__(self,
                             name=name,
                             torch_module=TaskFlowModule(self),
                             activation=CompositeActivation(self),
                             decoder=TaskFlowDecoder(self),
                             dropout_mc=dropout_mc)

    def has_children(self):
        return True

    def get_treelib_explainer(self):
        return TreeExplainer(self.get_name(), self.get_flow_func(), self.tasks)


class TaskFlowForDevelopment(TaskForDevelopment, TaskFlowBase):

    def __init__(self, name: str, labels, inputs, tasks: Iterable[TaskForDevelopment], flow_func):
        TaskFlowBase.__init__(self, name, tasks, flow_func)
        TaskForDevelopment.__init__(self,
                                    name=name,
                                    labels=labels,
                                    criterion=TaskFlowCriterion(self, ctx=self.ctx),
                                    per_sample_criterion=None,
                                    available_func=None,
                                    metrics=self.get_metrics())
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

    def get_per_sample_criterion(self, prefix='', ctx=None):
        if ctx is None:
            ctx = self.ctx
        return TaskFlowLossPerSample(self, prefix=prefix, ctx=ctx)

    def get_metrics(self):
        all_metrics = []
        for task in self.tasks.values():
            all_metrics += task.get_metrics()
        return all_metrics

    def get_dataset(self, **kwargs) -> Dataset:
        return FlowDataset(self, **kwargs)

    def get_labels(self):
        all_labels = []
        for task in self.tasks.values():
            all_labels += task.get_labels()
        return all_labels

    def get_filter(self):
        return FilterCompositeVisitor(self, prefix='')

    def get_evaluator(self):
        return EvaluationCompositeVisitor(self, prefix='')


class UsedTasksTracer:

    def __init__(self):
        self.used_tasks = []

    def __getattr__(self, item):
        self.used_tasks.append(item)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __add__(self, other):
        return self

    def __invert__(self):
        return self

    def __or__(self, other):
        return self


def trace_used_tasks(flow_func, flow_name, name_to_task_dict):
    flow_name = flow_func.__name__ if flow_name is None else flow_name
    used_tasks_tracer = UsedTasksTracer()
    flow_func(used_tasks_tracer, UsedTasksTracer(), UsedTasksTracer())
    used_tasks = []
    for used_task_name in used_tasks_tracer.used_tasks:
        task = name_to_task_dict.get(used_task_name, None)
        if task is not None:
            used_tasks.append(task)
    return flow_name, used_tasks


class Tasks:
    """
    Represents a collections of related tasks and task flows.
    """

    def __init__(self, leaf_tasks: List[MinimalTask]):
        self.leaf_tasks = leaf_tasks
        self.flow_tasks = []
        self._name_to_task = {}
        for leaf_task in self.leaf_tasks:
            self._name_to_task[leaf_task.get_name()] = leaf_task

    def add_task_flow(self, task_flow: TaskFlowMinimal) -> TaskFlowMinimal:
        self.flow_tasks.append(task_flow)
        self._name_to_task[task_flow.get_name()] = task_flow
        return task_flow

    def add_flow(self, func, flow_name=None, dropout_mc=None) -> TaskFlowMinimal:
        flow_name = func.__name__ if flow_name is None else flow_name
        flow = self.create_flow(func, flow_name, dropout_mc)
        return self.add_task_flow(flow)

    def create_flow(self, flow_func, flow_name=None, dropout_mc=None) -> TaskFlowMinimal:
        name_to_task_dict = self._name_to_task
        flow_name, used_tasks = trace_used_tasks(flow_func, flow_name, name_to_task_dict)
        return TaskFlowMinimal(name=flow_name,
                               tasks=used_tasks,
                               flow_func=flow_func,
                               dropout_mc=dropout_mc)

    def get_all_tasks(self) -> List[MinimalTask]:
        return self.leaf_tasks + self.flow_tasks

    def get_full_flow(self) -> TaskFlowMinimal:
        return self.flow_tasks[-1]

    def get_task(self, task_name) -> MinimalTask:
        return self._name_to_task.get(task_name)
