from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Mapping

import numpy as np
import torch
from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder, State, IRunner
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, SequentialSampler

from dnn_cool.task_flow import TaskFlow
from dnn_cool.utils import any_value


def publish_all(prefix: str,
                idx: int,
                sample: Tuple[Dict, Dict],
                mapping_key: str,
                key: str,
                writer: SummaryWriter,
                mapping: Mapping,
                task_name: str):
    """
    Publishes a given key given all publishers supplied via the mapping
    :param prefix: The prefix, this is typically either "best" or "worst".

    :param idx: The index of the sample in the dataset.

    :param sample: A tuple X, y where X and y and dictionaries.

    :param mapping_key: The key which when applied to the mapping gives a list of publishers.

    :param key: The key in X that is being published.

    :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

    :param mapping: A mapping `[str, List[Callable]]` where the key are either input columns, or input types and the values are a list of publisher functions.

    :param task_name: The name of the task, to be included in the name
    """
    if mapping_key in mapping:
        publishers = mapping[mapping_key]
        for publisher in publishers:
            publisher(writer, idx, sample, prefix, task_name, key)


def img(writer: SummaryWriter, idx: int, sample: Tuple[Dict, Dict], prefix: str, task_name: str, key: str):
    """
    Publishes image interpretation data.

    :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

    :param idx: The index of the dataset sample.

    :param sample: A tuple of X, y dictionaries.

    :param prefix: The prefix, this is typically either "best" or "worst".

    :param task_name: The name of the task, to be included in the name.

    :param key: The key in X that is being published.
    """
    X, y = sample
    writer.add_image(f'{prefix}_{task_name}_images', X[key])


def text(writer: SummaryWriter, idx: int, sample: Tuple, prefix: str, task_name: str, key: str):
    """
    Publishes text interpretation data.

     :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

    :param idx: The index of the dataset sample.

    :param sample: A tuple of X, y dictionaries.

    :param prefix: The prefix, this is typically either "best" or "worst".

    :param task_name: The name of the task, to be included in the name.

    :param key: The key in X that is being published.
    """
    X, y = sample
    writer.add_text(f'{prefix}_{task_name}_text', X[key])


def default_tensorboard_type_mapping():
    return {
        'img': [img],
        'text': [text]
    }


@dataclass()
class TensorboardConverter:
    """
    A dataclass which holds mappings from column names to Tensorboard publishers and from column types to Tensorboard
    publishers. Also, it stores which column is of what type, to be able to log any column name to the Tensorboard.
    """
    col_mapping: Dict[str, List[Callable]] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column names to a list of publishers. A publisher is just a callable which will be called 
    with this signature: 
    :code:`publisher(writer: SummaryWriter, idx: int, sample: Tuple, prefix: str, task_name: str, key: str)`. Example 
    publisher functions are :meth:`dnn_cool.catalyst_utils.img` and :meth:`dnn_cool.catalyst_utils.text`.
    """
    type_mapping: Dict[str, List[Callable]] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column types to a list of publishers. A publisher is just a callable which will be called 
    with this signature: 
    :code:`publisher(writer: SummaryWriter, idx: int, sample: Tuple, prefix: str, task_name: str, key: str)`. Example 
    publisher functions are :meth:`dnn_cool.catalyst_utils.img` and :meth:`dnn_cool.catalyst_utils.text`.
    """
    col_to_type_mapping: Dict[str, str] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column names to type names.
    """

    def __call__(self, writer: SummaryWriter, idx: int, sample: Tuple[Dict, Dict], prefix: str, task_name: str):
        """
        Publishes a given sample to the tensorboard, using the mappings defined in the dataclass.

        :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

        :param idx: The index of the sample that is being published

        :param sample: A tuple of dictionaries X, y.

        :param prefix: The prefix, this is typically either "best" or "worst".

        :param task_name: The name of the task

        :return:
        """
        if task_name == 'gt':
            return
        X, y = sample
        for key in X:
            publish_all(prefix, idx, sample, key, key, writer, self.col_mapping, task_name)
        for key in X:
            if key in self.col_to_type_mapping:
                publish_all(prefix, idx, sample, self.col_to_type_mapping[key], key, writer, self.type_mapping, task_name)


@dataclass
class TensorboardConverters:
    """
    This class handles the logging to the Tensorboard of an interpretation for a task
    """
    logdir: Path
    datasets: Dict[str, Dataset]
    tensorboard_loggers: Callable = field(default_factory=TensorboardConverter)
    loggers: Dict[str, SummaryWriter] = field(default_factory=lambda: {})
    top_k: int = 10

    def initialize(self, state: State):
        """
        Initializes the tensorboard loggers.

        :param state: The state with which the callback is called.
        """
        if (self.logdir is not None) and (state.loader_name not in self.loggers):
            path = str(self.logdir / f"{state.loader_name}_log")
            writer = SummaryWriter(path)
            self.loggers[state.loader_name] = writer

    def publish(self, state: State, interpretations: Dict[str, torch.Tensor]):
        """
        Publishes all interpretations

        :param state: The state with which the callback is called.

        :param interpretations: A dict object from task name to loss values. Also additional keys are those prefixed \
        with "indices|{task_name}", which hold the corresponding indices in the original dataset for which the loss \
        items are computed.

        """
        for key, value in interpretations.items():
            if key.startswith('indices'):
                continue
            sorted_indices = value.argsort()
            best_indices = interpretations[f'indices|{key}'][sorted_indices][:self.top_k]
            worst_indices = interpretations[f'indices|{key}'][sorted_indices][-self.top_k:]
            writer: SummaryWriter = self.loggers[state.loader_name]
            dataset = self.datasets[state.loader_name]
            self._publish_inputs(best_indices, writer, dataset, prefix='best', key=key)
            self._publish_inputs(worst_indices, writer, dataset, prefix='worst', key=key)

    def _publish_inputs(self, best_indices, writer, dataset, prefix, key):
        for idx in best_indices:
            if self.tensorboard_loggers is not None:
                self.tensorboard_loggers(writer, idx, dataset[idx], prefix, key)

    def close(self, state: State):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


class InterpretationCallback(Callback):
    """
    This callback publishes best and worst images per task, according to the configuration supplied via the constructor.
    """
    def __init__(self, flow: TaskFlow,
                 tensorboard_converters: Optional[TensorboardConverters] = None,
                 loaders_to_skip=()):
        """
        :param flow: The task flow, which holds the per sample loss functions for every task.

        :param tensorboard_converters: A :class:`TensorboardConverters` object which is responsible for the Tensorboard
        logging settings.

        :param loaders_to_skip: Optional loaders to be skipped, for example because labels aren't available for them.
        """
        super().__init__(CallbackOrder.Metric)
        self.flow = flow
        self.loaders_to_skip = loaders_to_skip

        self.overall_loss = flow.get_per_sample_loss()
        self.leaf_losses = self.overall_loss.get_leaf_losses_per_sample()
        self.interpretations = {}
        self.loader_counts = {}

        self.tensorboard_converters = tensorboard_converters

    def _initialize_interpretations(self):
        interpretation_dict = {
            'overall': [],
            'indices|overall': []
        }
        for path in self.leaf_losses:
            interpretation_dict[path] = []
            interpretation_dict[f'indices|{path}'] = []
        return interpretation_dict

    def on_loader_start(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
            return
        if state.loader_name in self.loaders_to_skip:
            return
        self.interpretations[state.loader_name] = self._initialize_interpretations()
        self.loader_counts[state.loader_name] = 0

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.initialize(state)

    def on_batch_end(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
            return
        if state.loader_name in self.loaders_to_skip:
            return
        outputs = state.output['logits']
        targets = state.input['targets']
        overall_res = self.overall_loss(outputs, targets)
        start = self.loader_counts[state.loader_name]
        bs = len(any_value(outputs))

        for path, loss in overall_res.items():
            if path.startswith('indices'):
                continue
            self.interpretations[state.loader_name][path].append(loss.detach().cpu().numpy())
            ind_key = f'indices|{path}'
            indices = overall_res[ind_key] + start
            self.interpretations[state.loader_name][ind_key].append(indices.detach().cpu().numpy())
        self.loader_counts[state.loader_name] += bs

    def on_loader_end(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
            return
        if state.loader_name in self.loaders_to_skip:
            return
        self.interpretations[state.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.interpretations[state.loader_name].items()
        }

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.publish(state, self.interpretations[state.loader_name])

    def on_stage_end(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
            return
        if state.loader_name in self.loaders_to_skip:
            return
        if self.tensorboard_converters is not None:
            self.tensorboard_converters.close(state)


class DeviceReducingDataParallel(DataParallel):

    def __init__(self, module: nn.Module, task_flow, callbacks):
        super().__init__(module)
        tasks_dict = task_flow.get_all_children()
        self.full_paths = []
        for full_path, task in tasks_dict.items():
            if task.is_train_only():
                continue
            self.full_paths.append(full_path)
        self._reducing_func = partial(reduce_on_device,
                                      criterion=task_flow.get_loss(),
                                      per_sample_criterion=task_flow.get_per_sample_loss(),
                                      leaf_criterions=task_flow.get_loss().get_leaf_losses(),
                                      metrics=task_flow.get_loss().get_metrics())
        self.callbacks = callbacks

    def gather(self, outputs, output_device):
        device_reduced_results = []
        dct = {
            'gt': {'_targets': {}}
        }
        for full_path in self.full_paths:
            dct[full_path] = []
            dct[f'precondition|{full_path}'] = []

        for i in range(len(outputs)):
            r = self._reducing_func(outputs=outputs[i], targets=outputs[i]['gt']['_targets'])
            device_reduced_results.append(r)
            for full_path in self.full_paths:
                dct[full_path].append(outputs[i][full_path].detach().cpu().numpy())
                precondition_path = f'precondition|{full_path}'
                dct[precondition_path].append(outputs[i][precondition_path].detach().cpu().numpy())
                np_targets = outputs[i]['gt']['_targets'][full_path].detach().cpu().numpy()
                if full_path not in dct['gt']['_targets']:
                    dct['gt']['_targets'][full_path] = []
                dct['gt']['_targets'][full_path].append(np_targets)

        for callback in self.callbacks:
            callback.on_dataparallel_gather(dct)
        gathered = super().gather(device_reduced_results, output_device)
        return gathered


class ReplaceGatherCallback(Callback):

    def __init__(self, task_flow, on_gather_callbacks=None):
        super().__init__(CallbackOrder.External)
        if on_gather_callbacks is None:
            on_gather_callbacks = []
        self.task_flow = task_flow
        self.on_gather_callbacks = on_gather_callbacks

    def on_stage_start(self, runner: "IRunner"):
        if isinstance(runner.model, DataParallel):
            runner.model = DeviceReducingDataParallel(runner.model.module, self.task_flow, self.on_gather_callbacks)


def reduce_on_device(criterion,
                     per_sample_criterion,
                     leaf_criterions,
                     metrics,
                     outputs,
                     targets):
    loss = criterion(outputs, targets)
    any_tensor = any_value(targets)
    n = len(any_tensor)
    reduced = {
        '_device|overall|loss': loss,
        '_device|overall|_n': torch.tensor(n, dtype=any_tensor.dtype, device=any_tensor.device)
    }
    for metric_name, metric in metrics:
        path = f'{metric.prefix}{metric.task_name}'
        full_name = f'{path}|{metric_name}'
        metric_res = metric(outputs, targets)
        if isinstance(metric_res, dict):
            for key, value in metric_res.items():
                value = torch.as_tensor(value, dtype=any_tensor.dtype, device=any_tensor.device)
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                reduced[f'_device|{full_name}_{key}'] = value
        else:
            value = torch.as_tensor(metric_res, dtype=any_tensor.dtype, device=any_tensor.device)
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
            reduced[f'_device|{full_name}'] = value
        reduced[f'_device|{path}|_n'] = outputs[f'precondition|{metric.prefix}{metric.task_name}'].sum()
    per_sample_losses = per_sample_criterion(outputs, targets)
    for key, value in per_sample_losses.items():
        if key.startswith('indices'):
            value += (n * value.device.index)
        if len(value.shape) == 0:
            value = value.unsqueeze(0)
        reduced[f'_device|{key}|loss_per_sample'] = value
        if not key.startswith('indices') and key != 'overall':
            reduced[f'_device|{key}|_n'] = outputs[f'precondition|{key}'].sum()
    for path, leaf_loss in leaf_criterions.items():
        reduced[f'_device|{path}|loss'] = leaf_loss(outputs, targets).loss_items
        reduced[f'_device|{path}|_n'] = outputs[f'precondition|{path}'].sum()
    return reduced