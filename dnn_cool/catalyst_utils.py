import numpy as np

from catalyst.core import Callback, CallbackOrder, State

from dnn_cool.losses import LossFlowData
from dnn_cool.task_flow import TaskFlow


def to_numpy(tensor):
    return tensor.squeeze(dim=-1).detach().cpu().numpy()


class InterpretationCallback(Callback):
    def __init__(self, flow: TaskFlow):
        super().__init__(CallbackOrder.Metric)
        self.flow = flow

        self.overall_loss = flow.get_per_sample_loss()
        self.leaf_losses = self.overall_loss.get_leaf_losses()
        self.interpretations = self._initialize_interpretations()

    def _initialize_interpretations(self):
        res = {'overall': []}
        for leaf_loss in self.leaf_losses:
            path = leaf_loss.prefix + leaf_loss.task_name
            res[path] = []
        return res

    def on_loader_start(self, state: State):
        self.interpretations = self._initialize_interpretations()

    def on_batch_end(self, state: State):
        outputs = state.batch_out
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
