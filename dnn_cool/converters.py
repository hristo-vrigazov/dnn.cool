from dataclasses import dataclass
from typing import Optional

import torch

from dnn_cool.catalyst_utils import TensorboardConverters, TensorboardConverter
from dnn_cool.task_flow import BinaryClassificationTask, ClassificationTask


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

        raise KeyError(f'Cannot convert column "{col}" from dataframe, because there is no registered converter'
                       f' for its guessed type {guessed_type}.')


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


@dataclass
class Converters:
    type = TypeGuesser()
    values = ValuesConverter()
    task = TaskConverter()

    tensorboard_converters = TensorboardConverter()
