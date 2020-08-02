from pathlib import Path

import numpy as np
from catalyst.core import Callback, CallbackOrder, State
from catalyst.utils.tools.tensorboard import SummaryWriter

from dnn_cool.losses import LossFlowData
from dnn_cool.task_flow import TaskFlow


def to_numpy(tensor):
    return tensor.squeeze(dim=-1).detach().cpu().numpy()


class InterpretationCallback(Callback):
    def __init__(self, flow: TaskFlow, datasets, logdir=None, top_k=10):
        super().__init__(CallbackOrder.Metric)
        self.flow = flow

        self.overall_loss = flow.get_per_sample_loss()
        self.leaf_losses = self.overall_loss.get_leaf_losses()
        self.interpretations = self._initialize_interpretations()

        self.logdir = logdir
        self.top_k = top_k
        self.publish_to_tensorboard = logdir is not None
        if self.publish_to_tensorboard:
            self.logdir = Path(self.logdir)
            self.loggers = {}
            self.datasets = datasets

    def _initialize_interpretations(self):
        res = {'overall': []}
        for leaf_loss in self.leaf_losses:
            path = leaf_loss.prefix + leaf_loss.task_name
            res[path] = []
        return res

    def on_loader_start(self, state: State):
        self.interpretations = self._initialize_interpretations()

        """Prepare tensorboard writers for the current stage"""
        if (self.logdir is not None) and (state.loader_name not in self.loggers):
            writer = SummaryWriter(self.logdir / f"{state.loader_name}_log")
            self.loggers[state.loader_name] = writer

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

        if self.publish_to_tensorboard:
            for key, value in self.interpretations.items():
                sorted_indices = value.argsort()
                best_indices = sorted_indices[:self.top_k]
                worst_indices = sorted_indices[-self.top_k:]
                writer: SummaryWriter = self.loggers[state.loader_name]
                dataset = self.datasets[state.loader_name]
                self._publish_inputs(best_indices, writer, dataset, prefix='best')
                self._publish_inputs(worst_indices, writer, dataset, prefix='worst')

    def _publish_inputs(self, best_indices, writer, dataset, prefix):
        for idx in best_indices:
            # TODO: here, we have to display the input according to their type, we won't always have just an image!
            X, y = dataset[idx]
            img = X['img']
            writer.add_image(f'{prefix}_overall', img)

    def on_stage_end(self, state: State):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()

