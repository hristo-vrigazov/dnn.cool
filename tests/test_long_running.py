import tempfile
from collections import OrderedDict

from catalyst.dl import SupervisedRunner
from torch import optim
from torch.utils.data import DataLoader

from dnn_cool.catalyst_utils import img_publisher, text_publisher, ReplaceGatherCallback
from dnn_cool.runner import TrainingArguments
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation
from dnn_cool.task_flow import TaskFlow
from dnn_cool.utils import torch_split_dataset


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
            num_epochs=20,
        )

    print_any_prediction(criterion, model, nested_loaders, runner)


def test_synthetic_dataset():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='security_logs')
    flow: TaskFlow = project.get_full_flow()
    criterion = flow.get_loss()

    args = TrainingArguments(
        num_epochs=2,
        callbacks=[],
        loaders=nested_loaders,
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
    )

    runner.train(**args)

    print_any_prediction(criterion, model, nested_loaders, runner)


def test_synthetic_dataset_default_runner():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment', balance_dataparallel_memory=True)
    flow: TaskFlow = project.get_full_flow()
    criterion = flow.get_loss()
    assert len(criterion.get_metrics()) < 100, 'Callbacks are too many!'

    project.converters.tensorboard_converters.type_mapping['img'] = [img_publisher]
    project.converters.tensorboard_converters.type_mapping['text'] = [text_publisher]
    # runner.train(num_epochs=10, callbacks=runner.default_callbacks[:1])
    runner.train(num_epochs=10)

    early_stop_callback = runner.default_callbacks[-1]
    assert early_stop_callback.best_score >= 0, 'Negative loss function!'
    print_any_prediction(criterion, model, nested_loaders, runner)


def print_any_prediction(criterion, model, nested_loaders, runner):
    loader = nested_loaders['valid']
    X, y = next(iter(loader))
    X = runner.batch_to_model_device(X)
    y = runner.batch_to_model_device(y)
    model = model.eval()
    pred = model(X)
    res = criterion(pred, y)
    print(res.item())
    print(pred, y)