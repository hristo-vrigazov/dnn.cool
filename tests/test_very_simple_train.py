import tempfile
import pytest
import torch
import numpy as np

from catalyst.dl import SupervisedRunner
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import optim


class DummyDataset(Dataset):
    """
    The function is the following:
    2 * x,  if x > 0
    -11 * x, else
    """

    def __getitem__(self, item):
        X_raw = torch.randn(1).float()
        X = {
            'features': X_raw,
            'gt': {
                'is_positive': (X_raw > 0).bool(),
            }
        }

        if (X_raw > 0).item():
            return X, {
                'is_positive': (X_raw > 0).float(),
                'positive_func': X_raw * 2,
                'negative_func': torch.zeros_like(X_raw).float(),
            }
        return X, {
            'is_positive': (X_raw > 0).float(),
            'positive_func': torch.zeros_like(X_raw).float(),
            'negative_func': X_raw * -11,
        }

    def __len__(self):
        return 2 ** 10


class NestedDummyDataset(Dataset):
    """
    The function is the following:
    2 * x,  if x > 0
    -11 * x, else
    """

    def __getitem__(self, item):
        X_raw = torch.randn(1).float()
        X = {
            'features': X_raw,
            'gt': {
                'is_positive': (X_raw > 0).bool(),
                'positive_flow.is_positive': (X_raw > 0).bool(),
                'negative_flow.is_positive': (X_raw > 0).bool()
            }
        }

        if (X_raw > 0).item():
            return X, {
                'is_positive': (X_raw > 0).float(),
                'positive_flow.is_positive': (X_raw > 0).float(),
                'positive_flow.positive_func': X_raw * 2,
                'positive_flow.negative_func': torch.zeros_like(X_raw),
                'negative_flow.is_positive': (X_raw > 0).float(),
                'negative_flow.positive_func': torch.zeros_like(X_raw).float(),
                'negative_flow.negative_func': X_raw * -11,
            }
        return X, {
            'is_positive': (X_raw > 0).float(),
            'positive_flow.is_positive': (X_raw > 0).float(),
            'positive_flow.positive_func': torch.zeros_like(X_raw),
            'positive_flow.negative_func': torch.zeros_like(X_raw),
            'negative_flow.is_positive': (X_raw > 0).float(),
            'negative_flow.positive_func': torch.zeros_like(X_raw).float(),
            'negative_flow.negative_func': X_raw * -11,
        }

    def __len__(self):
        return 2 ** 10


@pytest.fixture()
def loaders():
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return {
        'train': train_loader,
        'valid': val_loader
    }


@pytest.fixture()
def nested_loaders():
    train_dataset = NestedDummyDataset()
    val_dataset = NestedDummyDataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return {
        'train': train_loader,
        'valid': val_loader
    }


def test_very_simple_train(simple_linear_pair, loaders):
    model, simple_nesting_linear = simple_linear_pair

    print(model)

    runner = SupervisedRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        runner.train(
            model=model,
            criterion=simple_nesting_linear.loss(parent_reduction='mean', child_reduction='none'),
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            loaders=loaders,
            logdir=tmp_dir,
            num_epochs=100,
        )

    loader = loaders['valid']
    X, y = next(iter(loader))
    X = runner._batch2device(X, next(model.parameters()).device)
    model = model.eval()

    pred = model(X)
    print(pred, y)


def test_very_simple_train_nested(simple_nesting_linear_pair, nested_loaders):
    model, simple_nesting_linear = simple_nesting_linear_pair

    print(model)

    runner = SupervisedRunner()
    criterion = simple_nesting_linear.loss(parent_reduction='mean', child_reduction='none')

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            loaders=nested_loaders,
            logdir=tmp_dir,
            num_epochs=100,
        )

    loader = nested_loaders['valid']
    X, y = next(iter(loader))
    X = runner._batch2device(X, next(model.parameters()).device)
    y = runner._batch2device(y, next(model.parameters()).device)
    model = model.eval()

    pred = model(X)
    res = criterion(pred, y)
    print(res.item())
    print(pred, y)

