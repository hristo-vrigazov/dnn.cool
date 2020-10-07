from dataclasses import dataclass, field
from typing import Tuple, Dict

import numpy as np

from dnn_cool.catalyst_utils import TensorboardConverter


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
    type_mapping = {}

    def guess(self, df, output_col):
        if output_col in self.type_mapping:
            return self.type_mapping[output_col]

        if str(df[output_col].dtype) == 'bool':
            return 'binary'

        # TODO: smart guessing for different types here
        if str(df[output_col].dtype).startswith('int'):
            return 'category'

    def state_dict(self):
        return self.type_mapping

    def load_state_dict(self, state_dict):
        self.type_mapping = state_dict


def extract_state_when_possible(mapping):
    out_mapping = {}
    for key, value in mapping.items():
        try:
            out_mapping[key] = value.state_dict()
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return out_mapping


def load_state_when_possible(mapping, state_dict):
    for key, value in mapping.items():
        try:
            mapping[key].load_state_dict(state_dict[key])
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return mapping


@dataclass()
class StatefulConverter:
    col_mapping: Dict = field(default_factory=lambda: {})
    type_mapping: Dict = field(default_factory=lambda: {})

    def state_dict(self):
        return {
            'col': extract_state_when_possible(self.col_mapping),
            'type': extract_state_when_possible(self.type_mapping)
        }

    def load_state_dict(self, state_dict):
        load_state_when_possible(self.col_mapping, state_dict['col'])
        load_state_when_possible(self.type_mapping, state_dict['type'])


@dataclass()
class ValuesConverter(StatefulConverter):

    def to_values(self, df, col, guessed_type):
        if col in self.col_mapping:
            converter = self.col_mapping[col]
            return Values([col], converter(df[col]), [guessed_type])
        if guessed_type in self.type_mapping:
            converter = self.type_mapping[guessed_type]
            return Values([col], [converter(df[col])], [guessed_type])

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

    def state_dict(self):
        return {
            'type': self.type.state_dict(),
            'values': self.values.state_dict(),
            'task': self.task.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.type.load_state_dict(state_dict['type'])
        self.values.load_state_dict(state_dict['values'])
        self.task.load_state_dict(state_dict['task'])
