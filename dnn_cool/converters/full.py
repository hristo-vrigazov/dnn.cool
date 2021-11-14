from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Iterable

import pandas as pd

from dnn_cool.catalyst_utils import TensorboardConverter
from dnn_cool.converters.base import TypeGuesser, assert_col_in_df
from dnn_cool.converters.task.base import TaskConverter
from dnn_cool.converters.values.base import ValuesConverter
from dnn_cool.tasks.development.task_flow import TaskFlowForDevelopment, convert_task_flow_for_development
from dnn_cool.tasks.task_flow import TaskFlow
from dnn_cool.utils.base import Values
from dnn_cool.verbosity import StatsRegistry, log, Verbosity


@dataclass
class Converters:
    project_dir: Path
    name: str = 'converters'

    type: TypeGuesser = field(default_factory=TypeGuesser)
    values: ValuesConverter = field(default_factory=ValuesConverter)
    task: TaskConverter = field(default_factory=TaskConverter)

    tensorboard_converters: TensorboardConverter = field(default_factory=TensorboardConverter)

    stats_registry: StatsRegistry = field(default_factory=StatsRegistry)

    def state_dict(self):
        return {
            'type': self.type.state_dict(),
            'values': self.values.state_dict(),
            'task': self.task.state_dict()
        }

    def dump_state_to_directory(self, converters_directory: Path):
        converters_directory.mkdir(exist_ok=True, parents=True)
        self.type.dump_state_to_directory(converters_directory / 'type')
        self.values.dump_state_to_directory(converters_directory / 'values')
        self.task.dump_state_to_directory(converters_directory / 'task')

    def load_state_from_directory(self, converters_directory: Path):
        self.type.load_state_from_directory(converters_directory / 'type')
        self.values.load_state_from_directory(converters_directory / 'values')
        self.task.load_state_from_directory(converters_directory / 'task')

    def connect_to_stats_registry(self, stats_registry):
        self.values.connect_to_stats_registry(stats_registry)
        self.task.connect_to_stats_registry(stats_registry)

    def create_values(self, df, output_col):
        log(f'Creating values from column "{output_col}" in dataframe ...')
        values_type = self.type.guess(df, output_col)
        values = self.values.to_values(df, output_col, values_type)
        return values

    def read_inputs(self, df, input_col):
        log(f'Reading inputs from dataframe...')
        if isinstance(input_col, str):
            values = self.create_values(df, input_col)
            return values

        keys = []
        values = []
        types = []
        for col_s in input_col:
            vals = self.create_values(df, col_s)
            keys.extend(vals.keys)
            values.extend(vals.values)
            types.extend(vals.types)

        return Values(keys=keys, values=values, types=types)

    def create_leaf_task(self, df, col, task):
        values = self.create_values(df, col)
        task = self.task.to_task(col, values.types[0], values.values[0], task)
        return task

    def create_leaf_tasks(self, df, col, task):
        if isinstance(col, str):
            leaf_task = self.create_leaf_task(df, col, task.get(col))
            return {
                leaf_task.get_name(): leaf_task
            }

        res = {}
        for col_s in col:
            leaf_task = self.create_leaf_task(df, col_s, task.get(col_s))
            res[leaf_task.get_name()] = leaf_task
        return res

    def create_inputs_and_leaf_tasks_from_df(self,
                                             df: pd.DataFrame,
                                             input_col: Union[str, Iterable[str]],
                                             output_col: Union[str, Iterable[str]],
                                             task_flow,
                                             verbosity: Verbosity = Verbosity.SILENT):
        converters = self
        assert_col_in_df(df, input_col)
        assert_col_in_df(df, output_col)
        project_dir = Path(self.project_dir)
        converters_directory = project_dir / self.name
        stats_registry = StatsRegistry(project_dir / 'stats_registry.pkl', verbosity)
        converters.connect_to_stats_registry(stats_registry)
        if converters_directory.exists():
            converters.load_state_from_directory(converters_directory)

        inputs = self.read_inputs(df, input_col)
        leaf_tasks = self.create_leaf_tasks(df, output_col, task_flow)

        for i in range(len(inputs.keys)):
            converters.tensorboard_converters.col_to_type_mapping[inputs.keys[i]] = inputs.types[i]
        if not converters_directory.exists():
            converters.dump_state_to_directory(converters_directory)

        return inputs, leaf_tasks

    def create_task_flow_for_development(self,
                                         df: pd.DataFrame,
                                         input_col: Union[str, Iterable[str]],
                                         output_col: Union[str, Iterable[str]],
                                         task_flow: TaskFlow,
                                         verbosity: Verbosity = Verbosity.SILENT) -> TaskFlowForDevelopment:
        inputs, tasks_for_development = self.create_inputs_and_leaf_tasks_from_df(df,
                                                                                  input_col,
                                                                                  output_col,
                                                                                  task_flow,
                                                                                  verbosity)
        return convert_task_flow_for_development(inputs, task_flow, tasks_for_development)
