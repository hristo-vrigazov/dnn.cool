import torch
import numpy as np

from torch import nn, optim
from dnn_cool.uncertainty import DropoutMC


def test_samples_of_correct_shape():
    batch_size = 32
    in_features = 64
    out_features = 16
    module = nn.Linear(in_features, out_features).eval()
    dropout_mc = DropoutMC()

    x = torch.randn((batch_size, in_features))
    prediction_shape = module(x).shape
    expected_shape = (dropout_mc.num_samples,) + prediction_shape
    samples = dropout_mc.create_samples(module, None, x)
    actual_shape = samples.shape

    assert actual_shape == expected_shape
    mean_preds = samples.mean(dim=0)
    assert mean_preds.shape == prediction_shape
    std_preds = samples.std(dim=0)
    assert std_preds.shape == prediction_shape


def test_uncertainty_goes_down_with_training():
    n_features = 16
    w = torch.randn(1, n_features)
    b = torch.randn(1, n_features)
    x = torch.randn((int(2e4), n_features))
    y = w * x + b
    dropout_mc = DropoutMC()
    model = nn.Linear(n_features, n_features, bias=True).eval()
    std_before_training = dropout_mc.create_samples(model, None, x).std(dim=0)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    ratios = []
    for i in range(100):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        print(f'Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        model.eval()
        samples = dropout_mc.create_samples(model, None, x)
        std_after_training = samples.std(dim=0)
        std_is_smaller_ratio = (std_after_training < std_before_training).float().mean()
        ratios.append(std_is_smaller_ratio)

    ratios = np.array(ratios)
    assert (ratios >= 0.5).mean() > 0.9
