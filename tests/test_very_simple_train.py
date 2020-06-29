import tempfile
import pytest
import torch

from catalyst.dl import SupervisedRunner
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import optim


def dummy_data():
    X = torch.randn(64, 1).float()
    y = X.clone()
    y[X > 0] = (X * 2)[X > 0]
    y[X <= 0] = (X * -11)[X <= 0]
    return X, y


class DummyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        X = {
            'features': self.X[item],
            'gt': {
                'is_positive': (self.y[item] > 0).bool(),
            }
        }

        y = {
            'is_positive': (self.y[item] > 0).float(),
            'regress_positive': self.X[item] * 2,
            'regress_negative': self.y[item] * -11
        }
        return X, y

    def __len__(self):
        return len(self.X)


@pytest.fixture()
def loaders():
    train_dataset = DummyDataset(*dummy_data())
    val_dataset = DummyDataset(*dummy_data())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    return {
        'train': train_loader,
        'valid': val_loader
    }


def test_very_simple_train(simple_nesting_linear, loaders):
    model = simple_nesting_linear.torch()

    print(model)

    runner = SupervisedRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        runner.train(
            model=model,
            criterion=simple_nesting_linear.loss(reduction='none'),
            optimizer=optim.Adam(model.parameters(), lr=1e-2),
            loaders=loaders,
            logdir=tmp_dir,
            num_epochs=20,
        )
