from typing import Union, Iterable, Optional, Dict, Callable

from dnn_cool.task_flow import TaskFlow


class TypeGuesser:
    pass


def _not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns:' \
           f'{df.columns.tolist()}'


def _assert_col_in_df(col, df):
    if isinstance(col, str):
        assert col in df, _not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, _not_found_error_message(col_s, df)


class Project:

    def __init__(self, df,
                 input_col: Union[str, Iterable[str]],
                 output_col: Union[str, Iterable[str]],
                 converters: Optional[Dict[str, Callable]] = None):
        _assert_col_in_df(input_col, df)
        _assert_col_in_df(output_col, df)

        self.__leaf_tasks = self.__create_leaf_tasks(output_col, converters)
        self.__flow_tasks = []

    def add_flow(self, flow_name, flow_func):
        flow = self.create_flow(flow_name, flow_func)
        self.__flow_tasks.append(flow)
        return flow

    def create_flow(self, flow_name, flow_function):
        return TaskFlow(flow_name, self.get_all_tasks(), flow_function)

    def get_all_tasks(self):
        return self.__leaf_tasks + self.__flow_tasks

    def __create_leaf_tasks(self, output_col, converters):
        return []
