from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder, State
from torch.utils.data import Dataset, SequentialSampler

from dnn_cool.task_flow import TaskFlow
from dnn_cool.utils import any_value


def publish_all(prefix, sample, mapping_key, key, writer, mapping, task_name):
    if mapping_key in mapping:
        publishers = mapping[mapping_key]
        for publisher in publishers:
            publisher(writer, sample, prefix, task_name, key)


def img(writer: SummaryWriter, sample: Tuple, prefix: str, task_name: str, key: str):
    X, y = sample
    writer.add_image(f'{prefix}_{task_name}_images', X[key])


def text(writer: SummaryWriter, sample: Tuple, prefix: str, task_name: str, key: str):
    X, y = sample
    writer.add_text(f'{prefix}_{task_name}_text', X[key])


def default_tensorboard_type_mapping():
    return {
        'img': [img],
        'text': [text]
    }


@dataclass()
class TensorboardConverter:
    col_mapping: Dict[str, List[Callable]] = field(default_factory=lambda: {})
    type_mapping: Dict[str, List[Callable]] = field(default_factory=default_tensorboard_type_mapping)
    col_to_type_mapping: Dict[str, str] = field(default_factory=lambda: {})

    def __call__(self, writer: SummaryWriter, sample: Tuple, prefix: str, task_name: str):
        if task_name == 'gt':
            return
        X, y = sample
        for key in X:
            publish_all(prefix, sample, key, key, writer, self.col_mapping, task_name)
        for key in X:
            if key in self.col_to_type_mapping:
                publish_all(prefix, sample, self.col_to_type_mapping[key], key, writer, self.type_mapping, task_name)


@dataclass
class TensorboardConverters:
    logdir: Path
    datasets: Dict[str, Dataset]
    tensorboard_loggers: Callable = field(default_factory=TensorboardConverter)
    loggers: Dict[str, SummaryWriter] = field(default_factory=lambda: {})
    top_k: int = 10

    def initialize(self, state):
        if (self.logdir is not None) and (state.loader_name not in self.loggers):
            path = str(self.logdir / f"{state.loader_name}_log")
            writer = SummaryWriter(path)
            self.loggers[state.loader_name] = writer

    def publish(self, state, interpretations):
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
                self.tensorboard_loggers(writer, dataset[idx], prefix, key)

    def close(self, state):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


class InterpretationCallback(Callback):
    def __init__(self, flow: TaskFlow, tensorboard_converters: Optional[TensorboardConverters] = None):
        super().__init__(CallbackOrder.Metric)
        self.flow = flow

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
        self.interpretations[state.loader_name] = self._initialize_interpretations()
        self.loader_counts[state.loader_name] = 0

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.initialize(state)

    def on_batch_end(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
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
        self.interpretations[state.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.interpretations[state.loader_name].items()
        }

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.publish(state, self.interpretations[state.loader_name])

    def on_stage_end(self, state: State):
        if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
            return
        if self.tensorboard_converters is not None:
            self.tensorboard_converters.close(state)
