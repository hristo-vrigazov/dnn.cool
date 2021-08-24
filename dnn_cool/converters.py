from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union, Iterable

import joblib
import pandas as pd

from dnn_cool.catalyst_utils import TensorboardConverter
from dnn_cool.tasks import TaskFlow, TaskFlowForDevelopment, convert_task_flow_for_development
from dnn_cool.utils import Values
from dnn_cool.verbosity import StatsRegistry, Logger, Verbosity, log


@dataclass()
class TypeGuesser:
    type_mapping: Dict = field(default_factory=lambda: {})

    def guess(self, df, output_col):
        if output_col in self.type_mapping:
            return self.type_mapping[output_col]

        if str(df[output_col].dtype) == 'bool':
            return 'binary'

        if str(df[output_col].dtype).startswith('int'):
            return 'category'

    def state_dict(self):
        return self.type_mapping

    def load_state_from_directory(self, directory: Path):
        mapping_file = directory / 'type_mapping.pkl'
        if not mapping_file.exists():
            return
        self.type_mapping = joblib.load(mapping_file)

    def dump_state_to_directory(self, directory):
        directory.mkdir(exist_ok=True)
        joblib.dump(self.type_mapping, directory / 'type_mapping.pkl')


def extract_state_when_possible(mapping):
    out_mapping = {}
    for key, value in mapping.items():
        try:
            out_mapping[key] = value.state_dict()
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return out_mapping


def load_state_from_directory_when_possible(mapping: Dict, directory: Path, stats_registry):
    for key, value in mapping.items():
        try:
            if (directory / f'{key}.pkl').exists():
                with Logger(f'Load "{directory}/{key}.pkl"', stats_registry=stats_registry):
                    if hasattr(value, 'load_from_system'):
                        mapping[key].load_from_system(directory, key)
                    else:
                        state_dict = joblib.load(directory / f'{key}.pkl')
                        mapping[key].load_state_dict(state_dict)
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return mapping


def dump_state_to_directory_when_possible(mapping, directory, stats_registry):
    for key, value in mapping.items():
        try:
            directory.mkdir(exist_ok=True)
            with Logger(f'Dump "{directory}/{key}.pkl"', stats_registry=stats_registry):
                if hasattr(value, 'dump_to_filesystem'):
                    value.dump_to_filesystem(directory, key)
                else:
                    joblib.dump(value.state_dict(), directory / f'{key}.pkl')
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass


@dataclass()
class StatefulConverter:
    col_mapping: Dict = field(default_factory=lambda: {})
    type_mapping: Dict = field(default_factory=lambda: {})
    stats_registry: StatsRegistry = field(default_factory=StatsRegistry)

    def state_dict(self):
        return {
            'col': extract_state_when_possible(self.col_mapping),
            'type': extract_state_when_possible(self.type_mapping)
        }

    def dump_state_to_directory(self, directory: Path):
        directory.mkdir(exist_ok=True)
        dump_state_to_directory_when_possible(self.col_mapping, directory / 'col', self.stats_registry)
        dump_state_to_directory_when_possible(self.type_mapping, directory / 'type', self.stats_registry)

    def load_state_from_directory(self, directory: Path):
        load_state_from_directory_when_possible(self.col_mapping, directory / 'col', self.stats_registry)
        load_state_from_directory_when_possible(self.type_mapping, directory / 'type', self.stats_registry)

    def connect_to_stats_registry(self, stats_registry):
        self.stats_registry = stats_registry


@dataclass()
class ValuesConverter(StatefulConverter):

    def to_values(self, df, col, guessed_type, perform_conversion=True):
        if col in self.col_mapping:
            converter = self.col_mapping[col]
            values = converter(df[col]) if perform_conversion else None
            return Values([col], [values], [guessed_type])
        if guessed_type in self.type_mapping:
            converter = self.type_mapping[guessed_type]
            values = converter(df[col]) if perform_conversion else None
            return Values([col], [values], [guessed_type])

        raise KeyError(f'Cannot convert column "{col}" from dataframe, because there is no registered converter'
                       f' for its guessed type "{guessed_type}".')


@dataclass()
class TaskConverter(StatefulConverter):

    def to_task(self, output_col, guessed_type, values):
        if output_col in self.col_mapping:
            return self.col_mapping[output_col](name=output_col, labels=values)

        return self.type_mapping[guessed_type](name=output_col, labels=values)


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def assert_col_in_df(df, col):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


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

    def create_leaf_task(self, df, col):
        values = self.create_values(df, col)
        task = self.task.to_task(col, values.types[0], values.values[0])
        return task

    def create_leaf_tasks(self, df, col):
        if isinstance(col, str):
            leaf_task = self.create_leaf_task(df, col)
            return {
                leaf_task.get_name(): leaf_task
            }

        res = {}
        for col_s in col:
            leaf_task = self.create_leaf_task(df, col_s)
            res[leaf_task.get_name()] = leaf_task
        return res

    def create_inputs_and_leaf_tasks_from_df(self,
                                             df: pd.DataFrame,
                                             input_col: Union[str, Iterable[str]],
                                             output_col: Union[str, Iterable[str]],
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
        leaf_tasks = self.create_leaf_tasks(df, output_col)

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
        inputs, tasks_for_development = self.create_inputs_and_leaf_tasks_from_df(df, input_col, output_col, verbosity)
        return convert_task_flow_for_development(inputs, task_flow, tasks_for_development)


