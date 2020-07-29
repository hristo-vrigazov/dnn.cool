import tempfile
import pytest
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import torch
from catalyst.dl import SupervisedRunner
from catalyst.utils import load_checkpoint, unpack_checkpoint
from torch import optim, nn
from torch.utils.data import DataLoader

from dnn_cool.project import Project, TypeGuesser, ValuesConverter, TaskConverter
from dnn_cool.synthetic_dataset import create_df_and_images_tensor
from dnn_cool.task_flow import TaskFlow, BoundedRegressionTask, BinaryClassificationTask
from dnn_cool.value_converters import binary_value_converter


def test_passenger_example(interior_car_task):
    model, task_flow = interior_car_task

    datasets = task_flow.get_dataset()
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

    @project.add_flow
    def camera_not_blocked_flow(flow, x, out):
        out += flow.door_open(x.features)
        out += flow.uniform_type(x.features) | out.door_open
        return out

    @project.add_flow
    def all_pipeline(flow, x, out):
        out += flow.camera_blocked(x.features)
        out += flow.camera_not_blocked_flow(x.features) | out.camera_blocked
        return out

    flow: TaskFlow = project.get_full_flow()
    print(flow)

    dataset = flow.get_dataset()
    print(dataset[0])

@pytest.fixture
def synthenic_dataset_preparation():
    imgs, df = create_df_and_images_tensor()
    output_col = ['camera_blocked', 'door_open', 'person_present', 'door_locked',
                  'face_x1', 'face_y1', 'face_w', 'face_h',
                  'body_x1', 'body_y1', 'body_w', 'body_h']
    type_guesser = TypeGuesser()
    type_guesser.type_mapping['camera_blocked'] = 'binary'
    type_guesser.type_mapping['door_open'] = 'binary'
    type_guesser.type_mapping['person_present'] = 'binary'
    type_guesser.type_mapping['door_locked'] = 'binary'
    type_guesser.type_mapping['face_x1'] = 'continuous'
    type_guesser.type_mapping['face_y1'] = 'continuous'
    type_guesser.type_mapping['face_w'] = 'continuous'
    type_guesser.type_mapping['face_h'] = 'continuous'
    type_guesser.type_mapping['body_x1'] = 'continuous'
    type_guesser.type_mapping['body_y1'] = 'continuous'
    type_guesser.type_mapping['body_w'] = 'continuous'
    type_guesser.type_mapping['body_h'] = 'continuous'
    type_guesser.type_mapping['img'] = 'img'
    values_converter = ValuesConverter()
    imgs /= 255
    values_converter.type_mapping['img'] = lambda x: imgs
    values_converter.type_mapping['binary'] = binary_value_converter

    def bounded_regression_converter(values):
        values = values.astype(float) / 64
        values[np.isnan(values)] = -1
        return torch.tensor(values).float().unsqueeze(dim=-1)

    def bounded_regression_decoder(values):
        return values * 64

    def bounded_regression_task():
        return partial(BoundedRegressionTask, module=nn.Linear(256, 1), decoder=bounded_regression_decoder)

    values_converter.type_mapping['continuous'] = bounded_regression_converter
    task_converter = TaskConverter()
    task_converter.col_mapping['camera_blocked'] = partial(BinaryClassificationTask, module=nn.Linear(256, 1))
    task_converter.col_mapping['door_open'] = partial(BinaryClassificationTask, module=nn.Linear(256, 1))
    task_converter.col_mapping['person_present'] = partial(BinaryClassificationTask, module=nn.Linear(256, 1))
    task_converter.col_mapping['door_locked'] = partial(BinaryClassificationTask, module=nn.Linear(256, 1))
    task_converter.col_mapping['face_x1'] = bounded_regression_task()
    task_converter.col_mapping['face_y1'] = bounded_regression_task()
    task_converter.col_mapping['face_w'] = bounded_regression_task()
    task_converter.col_mapping['face_h'] = bounded_regression_task()
    task_converter.col_mapping['body_x1'] = bounded_regression_task()
    task_converter.col_mapping['body_y1'] = bounded_regression_task()
    task_converter.col_mapping['body_w'] = bounded_regression_task()
    task_converter.col_mapping['body_h'] = bounded_regression_task()
    project = Project(df, input_col='img', output_col=output_col,
                      type_guesser=type_guesser, values_converter=values_converter, task_converter=task_converter)

    @project.add_flow
    def face_regression(flow, x, out):
        out += flow.face_x1(x.features)
        out += flow.face_y1(x.features)
        out += flow.face_w(x.features)
        out += flow.face_h(x.features)
        return out

    @project.add_flow
    def body_regression(flow, x, out):
        out += flow.body_x1(x.features)
        out += flow.body_y1(x.features)
        out += flow.body_w(x.features)
        out += flow.body_h(x.features)
        return out

    @project.add_flow
    def person_regression(flow, x, out):
        out += flow.face_regression(x)
        out += flow.body_regression(x)
        return out

    @project.add_flow
    def full_flow(flow, x, out):
        out += flow.camera_blocked(x.features)
        out += flow.door_open(x.features) | (~out.camera_blocked)
        out += flow.door_locked(x.features) | (~out.door_open)
        out += flow.person_present(x.features) | out.door_open
        out += flow.person_regression(x) | out.person_present
        return out

    flow: TaskFlow = project.get_full_flow()
    dataset = flow.get_dataset()
    module = flow.torch()
    print(module)
    train_dataset = dataset
    val_dataset = dataset
    train_loader = DataLoader(train_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=False)
    nested_loaders = OrderedDict({
        'train': train_loader,
        'valid': val_loader
    })
    runner = SupervisedRunner()
    criterion = flow.get_loss(parent_reduction='mean', child_reduction='none')
    callbacks = criterion.catalyst_callbacks()

    class SecurityModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.Conv2d(64, 128, kernel_size=5),
                nn.AvgPool2d(2),
                nn.Conv2d(128, 128, kernel_size=5),
                nn.Conv2d(128, 256, kernel_size=5),
                nn.AvgPool2d(2),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.flow_module = full_flow.torch()

        def forward(self, x):
            res = {}
            res['features'] = self.seq(x['img'])
            res['gt'] = x.get('gt')
            return self.flow_module(res)

    model = SecurityModule()
    return callbacks, criterion, model, nested_loaders, runner, flow, df


def test_synthetic_dataset(synthenic_dataset_preparation):
    callbacks, criterion, model, nested_loaders, runner, flow, df = synthenic_dataset_preparation

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optim.Adam(model.parameters(), lr=1e-4),
            loaders=nested_loaders,
            callbacks=callbacks,
            logdir='./security_logs',
            num_epochs=2,
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


def test_inference_synthetic(synthenic_dataset_preparation):
    callbacks, criterion, model, nested_loaders, runner, flow, df = synthenic_dataset_preparation
    dataset = flow.get_dataset()

    ckpt = load_checkpoint('/home/hvrigazov/dnn.cool/tests/security_logs/checkpoints/best_full.pth')
    unpack_checkpoint(ckpt, model)

    idx = 1
    X, y = dataset[idx]
    del X['gt']

    for key in X:
        if key == 'gt':
            continue
        X[key] = X[key].unsqueeze(dim=0)

    model = model.eval()
    res = model(X)

    treelib_explainer = flow.get_treelib_explainer()

    tree = treelib_explainer(res)
    tree.show()
    
    for key in res:
        res[key] = res[key].item()
    print(pd.Series(res))
    print(df.iloc[idx])
