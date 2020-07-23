from typing import Union, Iterable, Optional, Dict, Callable

from dnn_cool.task_flow import TaskFlow, BinaryClassificationTask
from dataclasses import dataclass


@dataclass()
class TypeGuesser:
    type_mapping = {}

    def guess(self, df, output_col):
        if output_col in self.type_mapping:
            return self.type_mapping

        if str(df[output_col].dtype) == 'bool':
            return 'binary'

        #TODO: smart guessing for different types here
        if str(df[output_col].dtype).startswith('int'):
            return 'category'


@dataclass()
class TypeToTaskConverter:
    type_mapping = {
        'binary': BinaryClassificationTask
    }

    def convert(self, df, output_col, guessed_type):
        return self.type_mapping[guessed_type](name=output_col, labels=df[output_col].values)


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def _assert_col_in_df(col, df):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


def create_leaf_task(df, output_col, type_guesser, converters):
    values_type = type_guesser.guess(df, output_col)
    return converters.convert(df, output_col, values_type)


def create_leaf_tasks(df, output_col, type_guesser, converters):
    if isinstance(output_col, str):
        return create_leaf_task(df, output_col, type_guesser, converters)

    res = []
    for col in output_col:
        res.append(create_leaf_task(df, col, type_guesser, converters))
    return res


class Project:

    def __init__(self, df,
                 input_col: Union[str, Iterable[str]],
                 output_col: Union[str, Iterable[str]],
                 converters: Optional[TypeToTaskConverter] = None,
                 type_guesser: Optional[TypeGuesser] = None):
        _assert_col_in_df(input_col, df)
        _assert_col_in_df(output_col, df)

        if converters is None:
            converters = TypeToTaskConverter()
        if type_guesser is None:
            type_guesser = TypeGuesser()

        self.__leaf_tasks = create_leaf_tasks(df, output_col, type_guesser, converters)
        self.__flow_tasks = []

    def add_flow(self, flow_name, flow_func):
        flow = self.create_flow(flow_name, flow_func)
        self.__flow_tasks.append(flow)
        return flow

    def create_flow(self, flow_name, flow_function):
        return TaskFlow(flow_name, self.get_all_tasks(), flow_function)

    def get_all_tasks(self):
        return self.__leaf_tasks + self.__flow_tasks
