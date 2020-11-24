import torch
import matplotlib.pyplot as plt

from torch import nn

from dnn_cool.rf import compute_receptive_field
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation


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


def test_rf_multi_input():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment', balance_dataparallel_memory=False)
    loader = nested_loaders['valid']
    X, y = next(iter(loader))
    X = runner.batch_to_model_device(X)
    y = runner.batch_to_model_device(y)

    class FeaturesModel(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            features = self.model.seq[:4](x['syn_img'])
            return {
                'features': features
            }

    model = FeaturesModel(model).eval()
    with torch.no_grad():
        res = model(X)
        out_mask = {key: torch.zeros_like(value).bool() for key, value in res.items()}
        out_mask['features'][:, :, 11, 11] = True

    rf_mask = compute_receptive_field(model, X, out_mask)

    nonzero_indices = (torch.nonzero(rf_mask['syn_img']))
    mins = nonzero_indices.min(dim=0).values
    maxs = nonzero_indices.max(dim=0).values + 1
    lengths = maxs - mins

    assert lengths[0].item() == loader.batch_size
    assert lengths[1].item() == 3
    assert lengths[2].item() == 9
    assert lengths[3].item() == 9


