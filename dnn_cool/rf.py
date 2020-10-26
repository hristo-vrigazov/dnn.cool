import torch

from torch import nn


class RFModel(nn.Module):

    def __init__(self, inputs, model):
        super().__init__()
        # TODO: handle dict input
        self.dummy = nn.Parameter(data=torch.ones_like(inputs))
        self.model = model

    def forward(self, x):
        return self.model(self.dummy * x)


def compute_receptive_field(model, inputs, out_mask, marker=5000., eps=1e-3):
    rf_model = RFModel(inputs, model)
    pred = rf_model(inputs)
    target = pred.detach().clone()
    target[out_mask] += marker
    mse = nn.MSELoss()
    loss = mse(pred, target)
    loss.backward()
    rf_mask = abs(rf_model.dummy.grad) > eps
    return rf_mask
