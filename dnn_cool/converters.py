from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import joblib

from dnn_cool.catalyst_utils import TensorboardConverter
from dnn_cool.verbosity import StatsRegistry, Logger, Verbosity


class Values:

    def __init__(self, keys, values, types):
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values
        self.types = types

    def __getitem__(self, item):
        res = {}
        for i, key in enumerate(self.keys):
            res[key] = self.values[i][item]
        return res

    def __len__(self):
        return len(self.values[0])


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

    def to_values(self, df, col, guessed_type, perform_conversion):
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


@dataclass
class Converters:
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
        converters_directory.mkdir(exist_ok=True)
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
