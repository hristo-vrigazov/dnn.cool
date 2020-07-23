import tempfile
from collections import OrderedDict

from catalyst.dl import SupervisedRunner
from torch.utils.data import DataLoader
from torch import optim


def test_passenger_example(interior_car_task):
    model, task_flow = interior_car_task

    datasets = task_flow.get_labels()
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
    criterion = task_flow.get_loss(parent_reduction='mean', child_reduction='none')
    callbacks = criterion.catalyst_callbacks()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            loaders=nested_loaders,
            callbacks=callbacks,
            logdir=tmp_dir,
            num_epochs=40,
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


def test_project_example(simple_df_project):

    def camera_not_blocked_flow(flow, x, out):
        out += flow.uniform_type(x.features)
        return out

    simple_df_project.add_flow('camera_blocked_flow', flow_func=camera_not_blocked_flow)
