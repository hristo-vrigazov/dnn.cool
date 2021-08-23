from pathlib import Path
from typing import Union, Iterable, List

import pandas as pd

from dnn_cool.catalyst_utils import TensorboardConverter
from dnn_cool.converters import Values, Converters
from dnn_cool.runner import DnnCoolSupervisedRunner
from dnn_cool.verbosity import log, StatsRegistry, Verbosity




class ProjectForDevelopment:

    def __init__(self, inputs: Values,
                 full_flow: TaskFlowForDevelopment,
                 project_minimal: ProjectMinimal,
                 tensorboard_converters: TensorboardConverter):
        self.project_minimal = project_minimal
        self.project_dir = Path(project_minimal.project_dir)
        self.project_dir.mkdir(exist_ok=True)

        self.inputs = inputs
        self.full_flow = full_flow
        self.tensorboard_converters = tensorboard_converters

    def get_full_flow(self) -> TaskForDevelopment:
        return self.full_flow

    def runner(self,
               model,
               runner_name,
               early_stop=True,
               balance_dataparallel_memory=False,
               train_test_val_indices=None):
        minimal_runner = self.project_minimal.runner(model, runner_name)
        return DnnCoolSupervisedRunner(self, minimal_runner.model,
                                       early_stop=early_stop,
                                       balance_dataparallel_memory=balance_dataparallel_memory,
                                       runner_name=minimal_runner.runner_name,
                                       train_test_val_indices=train_test_val_indices)

    @classmethod
    def from_converters(cls, project_minimal: ProjectMinimal,
                        input_col: Union[str, Iterable[str]],
                        output_col: Union[str, Iterable[str]],
                        df: pd.DataFrame,
                        converters: Converters,
                        converters_name: str,
                        verbosity: Verbosity = Verbosity.SILENT):
        assert df is not None, 'Cannot create inputs before set_data(df, converters, name) has been called.'
        assert_col_in_df(input_col, df)
        assert_col_in_df(output_col, df)
        converters_directory = project_minimal.project_dir / converters_name
        if converters is None:
            converters = Converters()
        stats_registry = StatsRegistry(project_minimal.project_dir / 'stats_registry.pkl', verbosity)
        converters.connect_to_stats_registry(stats_registry)
        if converters_directory.exists():
            converters.load_state_from_directory(converters_directory)

        inputs = read_inputs(df, input_col, converters, perform_conversion=True)
        leaf_tasks = create_leaf_tasks(df, output_col, converters, perform_conversion=True)

        for i in range(len(inputs.keys)):
            converters.tensorboard_converters.col_to_type_mapping[inputs.keys[i]] = inputs.types[i]
        if not converters_directory.exists():
            converters.dump_state_to_directory(converters_directory)

        return cls(inputs=inputs)


# class Project:
#
#     def __init__(self, df,
#                  input_col: Union[str, Iterable[str]],
#                  output_col: Union[str, Iterable[str]],
#                  project_dir: Union[str, Path],
#                  converters: Converters,
#                  verbosity: Verbosity = Verbosity.SILENT):
#         self.df = df
#         perform_conversion = df is not None
#         self.perform_conversion = perform_conversion
#         if perform_conversion:
#             assert self.df is not None
#             assert_col_in_df(input_col, df)
#             assert_col_in_df(output_col, df)
#
#         self.project_dir = Path(project_dir)
#         self.project_dir.mkdir(exist_ok=True)
#
#         self.stats_registry = StatsRegistry(self.project_dir / 'stats_registry.pkl', verbosity)
#         converters_directory = self.project_dir / 'converters'
#         if converters is None:
#             converters = Converters()
#         self.converters = converters
#         self.converters.connect_to_stats_registry(self.stats_registry)
#         if converters_directory.exists() and perform_conversion:
#             self.converters.load_state_from_directory(converters_directory)
#
#         self.inputs = read_inputs(df, input_col, converters, perform_conversion)
#         self.leaf_tasks = create_leaf_tasks(df, output_col, converters, perform_conversion)
#
#         self.flow_tasks = []
#
#         self._name_to_task = {}
#         for leaf_task in self.leaf_tasks:
#             self._name_to_task[leaf_task.get_name()] = leaf_task
#
#         for i in range(len(self.inputs.keys)):
#             self.converters.tensorboard_converters.col_to_type_mapping[self.inputs.keys[i]] = self.inputs.types[i]
#         if not converters_directory.exists():
#             self.converters.dump_state_to_directory(converters_directory)
#
#     def add_task_flow(self, task_flow: TaskFlow):
#         self.flow_tasks.append(task_flow)
#         self._name_to_task[task_flow.get_name()] = task_flow
#         return task_flow
#
#     def add_flow(self, func, flow_name=None):
#         flow_name = func.__name__ if flow_name is None else flow_name
#         flow = self.create_flow(func, flow_name)
#         return self.add_task_flow(flow)
#
#     def create_flow(self, flow_func, flow_name=None):
#         flow_name = flow_func.__name__ if flow_name is None else flow_name
#         used_tasks_tracer = UsedTasksTracer()
#         flow_func(used_tasks_tracer, UsedTasksTracer(), UsedTasksTracer())
#         used_tasks = []
#         for used_task_name in used_tasks_tracer.used_tasks:
#             task = self._name_to_task.get(used_task_name, None)
#             if task is not None:
#                 used_tasks.append(task)
#         return TaskFlow(flow_name, used_tasks, flow_func, self.inputs)
#
#     def get_all_tasks(self):
#         return self.leaf_tasks + self.flow_tasks
#
#     def get_full_flow(self) -> TaskFlow:
#         return self.flow_tasks[-1]
#
#     def get_task(self, task_name):
#         return self._name_to_task[task_name]
#
#     def runner(self, model,
#                early_stop=True,
#                balance_dataparallel_memory=False,
#                runner_name=None,
#                train_test_val_indices=None):
#         return DnnCoolSupervisedRunner(self, model,
#                                        early_stop=early_stop,
#                                        balance_dataparallel_memory=balance_dataparallel_memory,
#                                        runner_name=runner_name,
#                                        train_test_val_indices=train_test_val_indices,
#                                        perform_conversion=self.perform_conversion)
