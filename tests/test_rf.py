import torch
import matplotlib.pyplot as plt

from torch import nn

from dnn_cool.rf import compute_receptive_field


def test_rf_simple_conv():
    x = torch.randn(1, 1, 8, 8)
    out_mask = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    out_mask[:, :, 0, 0] = True

    model = nn.Conv2d(1, 1, (3, 3))
    rf_mask = compute_receptive_field(model, x, out_mask)

    plt.matshow(rf_mask[0, 0])
    plt.show()

    plt.matshow(out_mask[0, 0])
    plt.show()


def test_rf_simple_summing():
    x = torch.randn(1, 8)
    out_mask = torch.zeros(1, 2, dtype=torch.bool)
    out_mask[:, 0] = True

    class SimpleModule(nn.Module):

        def forward(self, x):
            a = x[:, :6].sum(dim=1)
            b = x[:, 6:].sum(dim=1)
            return torch.cat((a, b)).unsqueeze(0)

    model = SimpleModule()
    rf_mask = compute_receptive_field(model, x, out_mask)

    plt.matshow(rf_mask)
    plt.show()

    plt.matshow(out_mask)
    plt.show()

