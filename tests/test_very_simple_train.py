import tempfile
from collections import OrderedDict

from catalyst.dl import SupervisedRunner
from torch.utils.data import DataLoader
from torch import optim

from dnn_cool.project import Project
from dnn_cool.task_flow import TaskFlow

import pandas as pd


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


def test_project_example():
    df_data = [
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4},
    ]

    df = pd.DataFrame(df_data)

    project = Project(df, input_col='input', output_col=['camera_blocked', 'door_open', 'uniform_type'])

    def camera_not_blocked_flow(flow, x, out):
        out += flow.camera_blocked(x.features)
        out += flow.door_open(x.features) | (~out.camera_blocked)
        out += flow.uniform_type(x.features) | out.door_open
        return out

    project.add_flow(camera_not_blocked_flow)

    flow: TaskFlow = project.get_full_flow()
    print(flow)

    dataset = flow.get_labels()
    print(dataset[0])
