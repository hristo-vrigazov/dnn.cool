from typing import Iterable, Optional, Callable, Tuple, List, Dict

import torch
from torch import nn
from treelib import Tree

from dnn_cool.activations import CompositeActivation
from dnn_cool.utils import Values
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


class IMinimal:

    def get_minimal(self):
        raise NotImplementedError()


class Task(IMinimal):

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

    def get_minimal(self):
        return self


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


class BinaryHardcodedTask(Task):

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


class BoundedRegressionTask(Task):

    def __init__(self, name, torch_module, scale, dropout_mc=None):
        super().__init__(name, torch_module=torch_module, activation=nn.Sigmoid(),
                         decoder=BoundedRegressionDecoder(scale=scale),
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


class BinaryClassificationTask(Task):

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


class ClassificationTask(Task):

    def __init__(self, name, torch_module,
                 class_names: Optional[List[str]] = None,
                 top_k: Optional[int] = 5,
                 dropout_mc=None):
        super().__init__(name,
                         torch_module=torch_module,
                         activation=nn.Softmax(dim=-1),
                         decoder=ClassificationDecoder(),
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


class TaskFlowBase:

    def __init__(self, name, tasks, flow_func):
        self.name = name
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task
        self.flow_func = flow_func
        self.ctx = {}

    def get_name(self):
        return self.name

    def get_flow_func(self):
        return self.flow_func

    def get_all_children(self, prefix=''):
        tasks = {}
        for task_name, task in self.tasks.items():
            if task.get_minimal().has_children():
                assert isinstance(task, TaskFlowBase)
                tasks.update(task.get_all_children(prefix=f'{prefix}{task.get_name()}.'))
            else:
                tasks[prefix + task_name] = task
        return tasks


class TaskFlow(Task, TaskFlowBase):

    def __init__(self, name, tasks: Iterable[Task], flow_func, dropout_mc=None):
        TaskFlowBase.__init__(self, name, tasks, flow_func)
        Task.__init__(self,
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

    def __init__(self, leaf_tasks: List[Task]):
        self.leaf_tasks = leaf_tasks
        self.flow_tasks = []
        self.task_dict = {}
        for leaf_task in self.leaf_tasks:
            self.task_dict[leaf_task.get_name()] = leaf_task

    def add_task_flow(self, task_flow: TaskFlow) -> TaskFlow:
        self.flow_tasks.append(task_flow)
        self.task_dict[task_flow.get_name()] = task_flow
        return task_flow

    def add_flow(self, func, flow_name=None, dropout_mc=None) -> TaskFlow:
        flow_name = func.__name__ if flow_name is None else flow_name
        flow = self.create_flow(func, flow_name, dropout_mc)
        return self.add_task_flow(flow)

    def create_flow(self, flow_func, flow_name=None, dropout_mc=None) -> TaskFlow:
        name_to_task_dict = self.task_dict
        flow_name, used_tasks = trace_used_tasks(flow_func, flow_name, name_to_task_dict)
        return TaskFlow(name=flow_name,
                        tasks=used_tasks,
                        flow_func=flow_func,
                        dropout_mc=dropout_mc)

    def get_all_tasks(self) -> Dict[str, Task]:
        return self.task_dict

    def get_full_flow(self) -> TaskFlow:
        return self.flow_tasks[-1]


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
