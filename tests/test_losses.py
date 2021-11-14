import torch
from torch import nn

from dnn_cool.losses.torch import ReducedPerSample
from dnn_cool.losses.base import CriterionFlowData


def test_loss_flow_data_creation():
    outputs = {
        'out_task0': torch.ones(8),
        'out_task1': torch.ones(8)
    }
    targets = {
        'target_task0': torch.zeros(8),
        'target_task1': torch.zeros(8)
    }
    loss_flow_data = CriterionFlowData.from_args(outputs, targets)
    assert loss_flow_data is not None


def test_loss_flow_data_creation_from_other():
    outputs = {
        'out_task0': torch.ones(8),
        'out_task1': torch.ones(8)
    }
    targets = {
        'target_task0': torch.zeros(8),
        'target_task1': torch.zeros(8)
    }
    loss_flow_data = CriterionFlowData(outputs, targets)
    loss_flow_data = CriterionFlowData.from_args(loss_flow_data, loss_flow_data)
    assert loss_flow_data is not None
