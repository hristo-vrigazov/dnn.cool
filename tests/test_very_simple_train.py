import tempfile
from collections import OrderedDict

from catalyst.dl import SupervisedRunner
from torch.utils.data import DataLoader
from torch import optim


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


def test_very_simple_train_nested(simple_nesting_linear_pair):
    model, simple_nesting_linear = simple_nesting_linear_pair

    datasets = simple_nesting_linear.datasets()
    # TODO: use train/test split
    train_dataset = datasets
    val_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    nested_loaders = OrderedDict({
        'train': train_loader,
        'valid': val_loader
    })

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

