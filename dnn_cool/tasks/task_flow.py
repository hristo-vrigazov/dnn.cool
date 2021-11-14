from typing import Iterable, List, Dict

from dnn_cool.activations import CompositeActivation
from dnn_cool.decoders.base import TaskFlowDecoder
from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.help import helper
from dnn_cool.modules.torch import TaskFlowModule
from dnn_cool.tasks.base import Task
from dnn_cool.treelib import TreeExplainer


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

    def get(self, full_path):
        components = full_path.split('.')
        tmp = self
        for i in range(len(components) - 1):
            tmp = tmp.tasks[components[i]]
        return tmp.tasks[components[-1]]


class TaskFlow(Task, TaskFlowBase):

    def __init__(self, name, tasks: Iterable[Task], flow_func, autograd, dropout_mc=None):
        TaskFlowBase.__init__(self, name, tasks, flow_func)
        Task.__init__(self,
                      name=name,
                      torch_module=TaskFlowModule(self),
                      activation=CompositeActivation(self, prefix='', autograd=autograd),
                      decoder=TaskFlowDecoder(self, prefix='', autograd=autograd),
                      dropout_mc=dropout_mc)

    def has_children(self):
        return True

    def get_treelib_explainer(self):
        return TreeExplainer(self.get_name(), self.get_flow_func(), self.tasks)


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

    @helper(after_type='tasks')
    def __init__(self, leaf_tasks: List[Task], autograd=TorchAutoGrad()):
        self.leaf_tasks = leaf_tasks
        self.flow_tasks = []
        self.task_dict = {}
        for leaf_task in self.leaf_tasks:
            self.task_dict[leaf_task.get_name()] = leaf_task
        self.autograd = autograd

    def add_task_flow(self, task_flow: TaskFlow) -> TaskFlow:
        self.flow_tasks.append(task_flow)
        self.task_dict[task_flow.get_name()] = task_flow
        return task_flow

    @helper(after_type='tasks.add_flow')
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
                        dropout_mc=dropout_mc,
                        autograd=self.autograd)

    def get_all_tasks(self) -> Dict[str, Task]:
        return self.task_dict

    def get_full_flow(self) -> TaskFlow:
        return self.flow_tasks[-1]
