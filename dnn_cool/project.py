from typing import Union, Iterable, Optional, Dict, Callable

from dnn_cool.task_flow import TaskFlow, BinaryClassificationTask, ClassificationTask
from dataclasses import dataclass

import torch


class Values:

    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values

    def __getitem__(self, item):
        res = {}
        for i, key in enumerate(self.keys):
            res[key] = self.values[i][item]
        return res

    def __len__(self):
        return len(self.values[0])


@dataclass()
class TypeGuesser:
    type_mapping = {}

    def guess(self, df, output_col):
        if output_col in self.type_mapping:
            return self.type_mapping[output_col]

        if str(df[output_col].dtype) == 'bool':
            return 'binary'

        # TODO: smart guessing for different types here
        if str(df[output_col].dtype).startswith('int'):
            return 'category'


@dataclass()
class ValuesConverter:
    type_mapping = {}

    def to_values(self, df, col, guessed_type):
        if guessed_type in self.type_mapping:
            converter = self.type_mapping[guessed_type]
            return Values([col], [converter(df[col].values)])

        return Values([col], [torch.tensor(df[col].values)])


@dataclass()
class TaskConverter:
    col_mapping = {}
    type_mapping = {
        'binary': BinaryClassificationTask,
        'category': ClassificationTask
    }

    def to_task(self, output_col, guessed_type, values):
        if output_col in self.col_mapping:
            return self.col_mapping[output_col](name=output_col, labels=values)

        return self.type_mapping[guessed_type](name=output_col, labels=values)


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def assert_col_in_df(col, df):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


def create_values(df, output_col, type_guesser, values_converter):
    values_type = type_guesser.guess(df, output_col)
    values = values_converter.to_values(df, output_col, values_type)
    return values, values_type


def create_leaf_task(df, col, type_guesser, values_converter, task_converter):
    values, values_type = create_values(df, col, type_guesser, values_converter)
    task = task_converter.to_task(col, values_type, values.values[0])
    return task


def create_leaf_tasks(df, col, type_guesser, values_converter, task_converter):
    if isinstance(col, str):
        return [create_leaf_task(df, col, type_guesser, values_converter, task_converter)]

    res = []
    for col_s in col:
        res.append(create_leaf_task(df, col_s, type_guesser, values_converter, task_converter))
    return res


def read_inputs(df, input_col, type_guesser, values_converter):
    if isinstance(input_col, str):
        values, values_type = create_values(df, input_col, type_guesser, values_converter)
        return values

    keys = []
    values = []
    for col_s in input_col:
        vals, _ = create_values(df, col_s, type_guesser, values_converter)
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
                 type_guesser: TypeGuesser = TypeGuesser(),
                 values_converter: ValuesConverter = ValuesConverter(),
                 task_converter: TaskConverter = TaskConverter()):
        assert_col_in_df(input_col, df)
        assert_col_in_df(output_col, df)

        self.inputs = read_inputs(df, input_col, type_guesser, values_converter)
        self.leaf_tasks = create_leaf_tasks(df, output_col, type_guesser, values_converter, task_converter)
        self.flow_tasks = []

        self._name_to_task = {}
        for leaf_task in self.leaf_tasks:
            self._name_to_task[leaf_task.get_name()] = leaf_task

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

    def get_full_flow(self):
        return self.flow_tasks[-1]
