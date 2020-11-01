import torch

from torch import nn


def dummy_tensor(inputs):
    return nn.Parameter(data=torch.ones_like(inputs))


def dummy_dict(inputs):
    res = nn.ParameterDict()
    for key, value in inputs.items():
        if key != 'gt' and not key.startswith('precondition'):
            res[key] = dummy_tensor(value)
    return res


class RFModel(nn.Module):

    def __init__(self, inputs, model):
        super().__init__()
        self.dummy = dummy_dict(inputs) if isinstance(inputs, dict) else dummy_tensor(inputs)
        self.model = model

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.model(self.dummy * x)
        for key, value in self.dummy.items():
            x[key] *= value
        return self.model(x)


def compute_receptive_field(model, inputs, out_mask, marker=5e8):
    rf_model = RFModel(inputs, model)
    pred = rf_model(inputs)
    target = mark_predictions(pred, out_mask, marker)
    rf_backprop(pred, target)
    rf_mask = get_rf_mask(rf_model)
    return rf_mask


def get_rf_mask(rf_model, eps=1e-4):
    if isinstance(rf_model.dummy, torch.Tensor):
        return abs(rf_model.dummy.grad) > eps
    rf_mask = {}
    for key, value in rf_model.dummy.items():
        rf_mask[key] = abs(value.grad) > eps
    return rf_mask


def rf_backprop(pred, target):
    if isinstance(pred, torch.Tensor):
        mse = nn.MSELoss()
        loss = mse(pred, target)
        loss.backward()
        return
    mse = nn.MSELoss()
    losses = []
    for key, value in pred.items():
        losses.append(mse(value, target[key]))
    loss = torch.stack(losses).mean()
    loss.backward()


def mark_predictions(pred, out_mask, marker):
    if isinstance(pred, torch.Tensor):
        target = pred.detach().clone()
        target[out_mask] += marker
        return target
    assert isinstance(pred, dict)
    target = {}
    for key, value in pred.items():
        target[key] = value.detach().clone()
        target[key][out_mask[key]] += marker
    return target
