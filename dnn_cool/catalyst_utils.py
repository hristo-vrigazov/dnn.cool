from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from catalyst.core import Callback, CallbackOrder, State
from catalyst.utils.tools.tensorboard import SummaryWriter
from dataclasses import dataclass, field

from torch.utils.data import Dataset

from dnn_cool.losses import LossFlowData
from dnn_cool.task_flow import TaskFlow


def to_numpy(tensor):
    return tensor.squeeze(dim=-1).detach().cpu().numpy()


class TensorboardConverter:

    def __call__(self, writer: SummaryWriter, sample: Tuple, prefix: str):
        X, y = sample
        writer.add_image(f'{prefix}_images', X['img'])


@dataclass
class TensorboardConverters:
    logdir: Path
    datasets: Dict[str, Dataset]
    tensorboard_loggers: Callable = TensorboardConverter()
    loggers: Dict[str, SummaryWriter] = field(default_factory=lambda: {})
    top_k: int = 10

    def initialize(self, state):
        if (self.logdir is not None) and (state.loader_name not in self.loggers):
            path = str(self.logdir / f"{state.loader_name}_log")
            writer = SummaryWriter(path)
            self.loggers[state.loader_name] = writer

    def publish(self, state, interpretations):
        for key, value in interpretations.items():
            sorted_indices = value.argsort()
            best_indices = sorted_indices[:self.top_k]
            worst_indices = sorted_indices[-self.top_k:]
            writer: SummaryWriter = self.loggers[state.loader_name]
            dataset = self.datasets[state.loader_name]
            self._publish_inputs(best_indices, writer, dataset, prefix=f'best_{key}')
            self._publish_inputs(worst_indices, writer, dataset, prefix=f'worst_{key}')

    def _publish_inputs(self, best_indices, writer, dataset, prefix):
        for idx in best_indices:
            if self.tensorboard_loggers is not None:
                self.tensorboard_loggers(writer, dataset[idx], prefix)

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
        self.leaf_losses = self.overall_loss.get_leaf_losses()
        self.interpretations = self._initialize_interpretations()

        self.tensorboard_converters = tensorboard_converters

    def _initialize_interpretations(self):
        res = {'overall': []}
        for leaf_loss in self.leaf_losses:
            path = leaf_loss.prefix + leaf_loss.task_name
            res[path] = []
        return res

    def on_loader_start(self, state: State):
        self.interpretations = self._initialize_interpretations()

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.initialize(state)

    def on_batch_end(self, state: State):
        outputs = state.batch_out['logits']
        targets = state.batch_in['targets']
        self.interpretations['overall'].append(to_numpy(self.overall_loss(outputs, targets)))

        for loss in self.leaf_losses:
            path = loss.prefix + loss.task_name
            loss_flow_data = LossFlowData(outputs, targets)
            loss_items = loss(loss_flow_data).loss_items
            self.interpretations[path].append(to_numpy(loss_items))

    def on_loader_end(self, state: State):
        self.interpretations = {
            key: np.concatenate(value, axis=0)
            for key, value in self.interpretations.items()
        }

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.publish(state, self.interpretations)

    def on_stage_end(self, state: State):
        if self.tensorboard_converters is not None:
            self.tensorboard_converters.close(state)

