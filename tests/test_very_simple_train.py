import tempfile
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from catalyst.dl import SupervisedRunner
from catalyst.utils import load_checkpoint, unpack_checkpoint
from torch import optim, nn
from torch.utils.data import DataLoader

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters, TensorboardConverter
from dnn_cool.converters import TypeGuesser, ValuesConverter, TaskConverter, Converters
from dnn_cool.project import Project
from dnn_cool.runner import InferDictCallback
from dnn_cool.synthetic_dataset import create_df_and_images_tensor, synthenic_dataset_preparation
from dnn_cool.task_flow import TaskFlow, BoundedRegressionTask, BinaryClassificationTask
from dnn_cool.utils import torch_split_dataset
from dnn_cool.value_converters import binary_value_converter


def test_passenger_example(interior_car_task):
    model, task_flow = interior_car_task

    dataset = task_flow.get_dataset()

    train_dataset, val_dataset = torch_split_dataset(dataset, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    nested_loaders = OrderedDict({
        'train': train_loader,
        'valid': val_loader
    })

    print(model)

    runner = SupervisedRunner()
    criterion = task_flow.get_loss()
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

    print_any_prediction(criterion, model, nested_loaders, runner)


def test_project_example():
    df_data = [
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4},
    ]

    df = pd.DataFrame(df_data)

    converters = Converters()
    converters.values.type_mapping['category'] = torch.LongTensor
    converters.values.type_mapping['binary'] = torch.BoolTensor

    project = Project(df, input_col='input',
                      output_col=['camera_blocked', 'door_open', 'uniform_type'],
                      project_dir='./example_project',
                      converters=converters)

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


def test_synthetic_dataset():
    callbacks, criterion, model, nested_loaders, runner, flow, df, val_dataset, project = synthenic_dataset_preparation()

    with tempfile.TemporaryDirectory() as tmp_dir:
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optim.Adam(model.parameters(), lr=1e-4),
            loaders=nested_loaders,
            callbacks=callbacks,
            logdir='./security_logs',
            num_epochs=2,
        )

    print_any_prediction(criterion, model, nested_loaders, runner)


def test_inference_synthetic():
    callbacks, criterion, model, nested_loaders, runner, flow, df, val_dataset, project = synthenic_dataset_preparation()
    dataset = flow.get_dataset()

    n = 4 * torch.cuda.device_count()
    loader = DataLoader(dataset, batch_size=n, shuffle=False)

    ckpt = load_checkpoint('/home/hvrigazov/dnn.cool/tests/security_project/security_logs/checkpoints/best_full.pth')
    unpack_checkpoint(ckpt, model)

    X, y = next(iter(loader))
    del X['gt']

    model = model.eval()
    res = model(X)

    treelib_explainer = flow.get_treelib_explainer()

    tree = treelib_explainer(res)
    tree.show()

    for i in range(n):
        img = (dataset[i][0]['img'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f'Img {i}')
        plt.show()


def test_interpretation_synthetic():
    callbacks, criterion, model, nested_loaders, runner, flow, df, datasets, project = synthenic_dataset_preparation()

    loaders = OrderedDict({'infer': nested_loaders['valid']})

    ckpt = load_checkpoint('/home/hvrigazov/dnn.cool/tests/security_project/security_logs/checkpoints/best_full.pth')
    unpack_checkpoint(ckpt, model)

    tensorboard_converters = TensorboardConverters(
        logdir=Path('./security_logs'),
        tensorboard_loggers=TensorboardConverter(),
        datasets=datasets
    )

    callbacks = OrderedDict([
        ("interpretation", InterpretationCallback(flow, tensorboard_converters)),
        ("inference", InferDictCallback())
    ])
    r = runner.infer(model,
                     loaders=loaders,
                     callbacks=callbacks,
                     logdir='./security_logs')

    interpretations = callbacks["interpretation"].interpretations
    print(interpretations)


def test_synthetic_dataset_default_runner():
    callbacks, criterion, model, nested_loaders, runner, flow, df, datasets, project = synthenic_dataset_preparation()

    runner.train(model=model, num_epochs=10)

    early_stop_callback = runner.default_callbacks[-1]
    assert early_stop_callback.best_score >= 0, 'Negative loss function!'
    print_any_prediction(criterion, model, nested_loaders, runner)


def test_interpretation_default_runner():
    callbacks, criterion, model, nested_loaders, runner, flow, df, datasets, project = synthenic_dataset_preparation()

    ckpt = load_checkpoint('/home/hvrigazov/dnn.cool/tests/security_project/security_logs/checkpoints/best_full.pth')
    unpack_checkpoint(ckpt, model)
    predictions, targets, interpretations = runner.infer(model=model)

    print(interpretations)
    print(predictions)


def test_full_pipeline():
    callbacks, criterion, model, nested_loaders, runner, flow, df, datasets, project = synthenic_dataset_preparation()

    ckpt = load_checkpoint('/home/hvrigazov/dnn.cool/tests/security_project/security_logs/checkpoints/best_full.pth')
    unpack_checkpoint(ckpt, model)
    predictions, targets, interpretations = runner.infer(model=model)

    print(interpretations)
    print(predictions)


def print_any_prediction(criterion, model, nested_loaders, runner):
    loader = nested_loaders['valid']
    X, y = next(iter(loader))
    X = runner._batch2device(X, next(model.parameters()).device)
    y = runner._batch2device(y, next(model.parameters()).device)
    model = model.eval()
    pred = model(X)
    res = criterion(pred, y)
    print(res.item())
    print(pred, y)
