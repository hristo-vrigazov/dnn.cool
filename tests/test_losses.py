import torch
from torch import nn

from dnn_cool.losses import ReducedPerSample, CriterionFlowData


def test_reduced_per_sample_utility(simple_binary_data):
    x, y, task = simple_binary_data

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    expected = criterion(x, y.unsqueeze(dim=1).float())

    x = x.repeat_interleave(3, dim=1).repeat_interleave(4, dim=1)
    y = y.unsqueeze(dim=-1).repeat_interleave(3, dim=1).repeat_interleave(4, dim=1).float()

    criterion = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), torch.mean)
    actual = criterion(x, y)

    assert torch.allclose(expected, actual)


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
