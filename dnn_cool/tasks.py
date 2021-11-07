from typing import Iterable, Optional, Callable, Tuple, List, Dict

import torch
from torch import nn
from treelib import Tree

from dnn_cool.datasets import FlowDataset
from dnn_cool.decoders.multilabel_classification import MultilabelClassificationDecoder
from dnn_cool.evaluation import EvaluationCompositeVisitor, EvaluationVisitor
from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.filter import FilterCompositeVisitor, FilterVisitor
from dnn_cool.losses import TaskFlowCriterion, ReducedPerSample, TaskFlowLossPerSample
from dnn_cool.metrics import TorchMetric, get_default_binary_metrics, \
    get_default_bounded_regression_metrics, get_default_classification_metrics, \
    get_default_multilabel_classification_metrics
from dnn_cool.missing_values import positive_values
from dnn_cool.modules import SigmoidAndMSELoss
from dnn_cool.tasks.base import IMinimal, Task
from dnn_cool.tasks.task_flow import TaskFlowBase, TaskFlow
from dnn_cool.utils import Values


class TaskForDevelopment(IMinimal):

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
        self.task = None

    def get_name(self) -> str:
        return self.name

    def get_filter(self) -> FilterVisitor:
        return FilterVisitor(self, prefix='')

    def get_evaluator(self) -> EvaluationVisitor:
        return EvaluationVisitor(self, prefix='')

    def get_available_func(self):
        return self.available_func

    def get_criterion(self, prefix='', ctx=None):
        return self.criterion

    def get_per_sample_criterion(self, prefix='', ctx=None):
        return self.per_sample_criterion

    def get_labels(self):
        return self.labels

    def get_metrics(self):
        for i in range(len(self.metrics)):
            metric_name, metric = self.metrics[i]
            metric.bind_to_task(self.task)
        return self.metrics

    def get_minimal(self):
        return self.task


class BinaryHardcodedTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        super().__init__(name,
                         labels=labels,
                         criterion=None,
                         per_sample_criterion=None,
                         available_func=positive_values,
                         metrics=[])


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


class BinaryClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        reduced_per_sample = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), reduction=torch.mean)
        super().__init__(name,
                         labels,
                         criterion=nn.BCEWithLogitsLoss(reduction='mean'),
                         per_sample_criterion=reduced_per_sample,
                         available_func=positive_values,
                         metrics=get_default_binary_metrics())


class ClassificationTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str,
                 labels):
        super().__init__(name,
                         labels,
                         criterion=nn.CrossEntropyLoss(reduction='mean'),
                         per_sample_criterion=ReducedPerSample(nn.CrossEntropyLoss(reduction='none'), torch.mean),
                         available_func=positive_values,
                         metrics=get_default_classification_metrics())


class MultilabelClassificationTask(Task):

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


class TaskFlowForDevelopment(TaskForDevelopment, TaskFlowBase):

    def __init__(self, name: str, inputs: Values, tasks: Iterable[TaskForDevelopment], flow_func: Callable,
                 labels=None):
        TaskFlowBase.__init__(self, name, tasks, flow_func)
        TaskForDevelopment.__init__(self,
                                    name=name,
                                    labels=labels,
                                    criterion=None,
                                    per_sample_criterion=None,
                                    available_func=None,
                                    metrics=self.get_metrics())
        self.inputs = inputs
        self.autograd = TorchAutoGrad()

    def get_inputs(self) -> Values:
        return self.inputs

    def get_criterion(self, prefix='', ctx=None):
        if ctx is None:
            ctx = self.ctx
        return TaskFlowCriterion(self, prefix=prefix, ctx=ctx)

    def get_per_sample_criterion(self, prefix='', ctx=None):
        if ctx is None:
            ctx = self.ctx
        return TaskFlowLossPerSample(self, prefix=prefix, ctx=ctx)

    def get_metrics(self):
        all_metrics = []
        for task in self.tasks.values():
            all_metrics += task.get_metrics()
        return all_metrics

    def get_dataset(self, **kwargs) -> FlowDataset:
        return FlowDataset(self, **kwargs)

    def get_labels(self):
        all_labels = []
        for task in self.tasks.values():
            all_labels += task.get_labels()
        return all_labels

    def get_filter(self):
        return FilterCompositeVisitor(self, prefix='', autograd=self.autograd)

    def get_evaluator(self):
        return EvaluationCompositeVisitor(self, prefix='', autograd=self.autograd)


def create_task_for_development(child: str,
                                inputs: Values,
                                minimal_tasks: Dict[str, TaskFlow],
                                tasks_for_development: Dict[str, TaskFlowForDevelopment]):
    task_for_development = tasks_for_development.get(child)
    if task_for_development is not None:
        task_for_development.task = minimal_tasks[child]
        return task_for_development
    task_flow = minimal_tasks[child]
    res = convert_task_flow_for_development(inputs, task_flow, tasks_for_development)
    return res


def convert_task_flow_for_development(inputs: Values,
                                      task_flow: TaskFlow,
                                      tasks_for_development: Dict[str, TaskFlowForDevelopment]):
    assert isinstance(task_flow, TaskFlow)
    full_flow_name = task_flow.flow_func.__name__
    child_tasks = []
    for child, child_task in task_flow.tasks.items():
        new_task = create_task_for_development(child, inputs, task_flow.tasks, tasks_for_development)
        child_tasks.append(new_task)
    res = TaskFlowForDevelopment(name=task_flow.get_name(),
                                 inputs=inputs,
                                 tasks=child_tasks,
                                 flow_func=task_flow.get_flow_func())
    res.task = task_flow
    tasks_for_development[full_flow_name] = res
    return res
