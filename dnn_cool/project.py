import torch

from pathlib import Path
from typing import Union, Iterable

from dnn_cool.converters import Values, Converters
from dnn_cool.runner import DnnCoolSupervisedRunner
from dnn_cool.task_flow import TaskFlow
from dnn_cool.verbosity import log, StatsRegistry, Verbosity


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def assert_col_in_df(col, df):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


def create_values(df, output_col, converters, perform_conversion):
    log(f'Creating values from column "{output_col}" in dataframe ...')
    values_type = converters.type.guess(df, output_col)
    values = converters.values.to_values(df, output_col, values_type, perform_conversion)
    return values


def create_leaf_task(df, col, converters, perform_conversion):
    values = create_values(df, col, converters, perform_conversion)
    task = converters.task.to_task(col, values.types[0], values.values[0])
    return task


def create_leaf_tasks(df, col, converters, perform_conversion):
    if isinstance(col, str):
        return [create_leaf_task(df, col, converters, perform_conversion)]

    res = []
    for col_s in col:
        res.append(create_leaf_task(df, col_s, converters, perform_conversion))
    return res


def read_inputs(df, input_col, converters, perform_conversion):
    log(f'Reading inputs from dataframe...')
    if isinstance(input_col, str):
        values = create_values(df, input_col, converters, perform_conversion)
        return values

    keys = []
    values = []
    types = []
    for col_s in input_col:
        vals = create_values(df, col_s, converters, perform_conversion)
        keys.extend(vals.keys)
        values.extend(vals.values)
        types.extend(vals.types)

    return Values(keys=keys, values=values, types=types)


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
                 converters: Converters,
                 verbosity: Verbosity = Verbosity.SILENT):
        self.df = df
        perform_conversion = df is not None
        self.perform_conversion = perform_conversion
        if perform_conversion:
            assert self.df is not None
            assert_col_in_df(input_col, df)
            assert_col_in_df(output_col, df)

        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)

        self.stats_registry = StatsRegistry(self.project_dir / 'stats_registry.pkl', verbosity)
        converters_directory = self.project_dir / 'converters'
        if converters is None:
            converters = Converters()
        self.converters = converters
        self.converters.connect_to_stats_registry(self.stats_registry)
        if converters_directory.exists() and perform_conversion:
            self.converters.load_state_from_directory(converters_directory)

        self.inputs = read_inputs(df, input_col, converters, perform_conversion)
        self.leaf_tasks = create_leaf_tasks(df, output_col, converters, perform_conversion)

        self.flow_tasks = []

        self._name_to_task = {}
        for leaf_task in self.leaf_tasks:
            self._name_to_task[leaf_task.get_name()] = leaf_task

        for i in range(len(self.inputs.keys)):
            self.converters.tensorboard_converters.col_to_type_mapping[self.inputs.keys[i]] = self.inputs.types[i]
        if not converters_directory.exists():
            self.converters.dump_state_to_directory(converters_directory)

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

    def runner(self, model,
               early_stop=True,
               balance_dataparallel_memory=False,
               runner_name=None,
               train_test_val_indices=None):
        return DnnCoolSupervisedRunner(self, model,
                                       early_stop=early_stop,
                                       balance_dataparallel_memory=balance_dataparallel_memory,
                                       runner_name=runner_name,
                                       train_test_val_indices=train_test_val_indices,
                                       perform_conversion=self.perform_conversion)
