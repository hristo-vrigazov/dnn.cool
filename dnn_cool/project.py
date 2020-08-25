from pathlib import Path
from typing import Union, Iterable

from dnn_cool.converters import Values, Converters
from dnn_cool.runner import DnnCoolSupervisedRunner
from dnn_cool.task_flow import TaskFlow


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def assert_col_in_df(col, df):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


def create_values(df, output_col, converters):
    values_type = converters.type.guess(df, output_col)
    values = converters.values.to_values(df, output_col, values_type)
    return values, values_type


def create_leaf_task(df, col, converters):
    values, values_type = create_values(df, col, converters)
    task = converters.task.to_task(col, values_type, values.values[0])
    return task


def create_leaf_tasks(df, col, converters):
    if isinstance(col, str):
        return [create_leaf_task(df, col, converters)]

    res = []
    for col_s in col:
        res.append(create_leaf_task(df, col_s, converters))
    return res


def read_inputs(df, input_col, converters):
    if isinstance(input_col, str):
        values, values_type = create_values(df, input_col, converters)
        return values

    keys = []
    values = []
    for col_s in input_col:
        vals, _ = create_values(df, col_s, converters)
        keys.extend(keys)
        values.extend(values)

    return Values(keys=keys, values=values)


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


class Project:

    def __init__(self, df,
                 input_col: Union[str, Iterable[str]],
                 output_col: Union[str, Iterable[str]],
                 project_dir: Union[str, Path],
                 converters: Converters = None):
        self.df = df
        assert_col_in_df(input_col, df)
        assert_col_in_df(output_col, df)

        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
        self.inputs = read_inputs(df, input_col, converters)
        self.leaf_tasks = create_leaf_tasks(df, output_col, converters)
        self.flow_tasks = []

        self._name_to_task = {}
        for leaf_task in self.leaf_tasks:
            self._name_to_task[leaf_task.get_name()] = leaf_task

        if converters is None:
            converters = Converters()
        self.converters = converters

    def add_task_flow(self, task_flow: TaskFlow):
        self.flow_tasks.append(task_flow)
        self._name_to_task[task_flow.get_name()] = task_flow
        return task_flow

    def add_flow(self, func, flow_name=None):
        flow_name = func.__name__ if flow_name is None else flow_name
        flow = self.create_flow(func, flow_name)
        return self.add_task_flow(flow)

    def create_flow(self, flow_func, flow_name=None):
        flow_name = flow_func.__name__ if flow_name is None else flow_name
        used_tasks_tracer = UsedTasksTracer()
        flow_func(used_tasks_tracer, UsedTasksTracer(), UsedTasksTracer())
        used_tasks = []
        for used_task_name in used_tasks_tracer.used_tasks:
            task = self._name_to_task.get(used_task_name, None)
            if task is not None:
                used_tasks.append(task)
        return TaskFlow(flow_name, used_tasks, flow_func, self.inputs)

    def get_all_tasks(self):
        return self.leaf_tasks + self.flow_tasks

    def get_full_flow(self) -> TaskFlow:
        return self.flow_tasks[-1]

    def get_task(self, task_name):
        return self._name_to_task[task_name]

    def runner(self, model, early_stop=True, runner_name=None, train_test_val_indices=None):
        return DnnCoolSupervisedRunner(self, model, early_stop, runner_name, train_test_val_indices)
