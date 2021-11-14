from typing import Iterable, Dict, Sequence

from dnn_cool.datasets import FlowDataset
from dnn_cool.evaluation import EvaluationCompositeVisitor
from dnn_cool.external.autograd import Tensor
from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.filter import FilterCompositeVisitor
from dnn_cool.losses.torch import TaskFlowCriterion, TaskFlowLossPerSample
from dnn_cool.tasks.development.base import TaskForDevelopment
from dnn_cool.tasks.task_flow import TaskFlowBase, TaskFlow
from dnn_cool.utils.base import Values, create_values_from_dict


class TaskFlowForDevelopment(TaskForDevelopment, TaskFlowBase):

    def __init__(self, task: TaskFlow,
                 tasks: Iterable[TaskForDevelopment],
                 inputs: Dict[str, Sequence[Tensor]] = None,
                 values: Values = None,
                 autograd=TorchAutoGrad(),
                 precondition_func=None,
                 labels=None):
        TaskFlowBase.__init__(self, task.get_name(), tasks, task.flow_func)
        TaskForDevelopment.__init__(self,
                                    task=task,
                                    labels=labels,
                                    criterion=None,
                                    per_sample_criterion=None,
                                    available_func=None,
                                    metrics=self.get_metrics(),
                                    autograd=autograd,
                                    precondition_func=precondition_func)
        self.inputs = values if values is not None else create_values_from_dict(inputs)
        self.autograd = autograd
        self.precondition_funcs = None

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

    def get_dataset(self) -> FlowDataset:
        return FlowDataset(self, precondition_funcs=self.precondition_funcs)

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
    res = TaskFlowForDevelopment(task=task_flow,
                                 values=inputs,
                                 tasks=child_tasks)
    res.task = task_flow
    tasks_for_development[full_flow_name] = res
    return res
