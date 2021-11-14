from catalyst.utils import any2device
from torch import optim

from dnn_cool.catalyst_utils import img_publisher, text_publisher
from dnn_cool.runner import TrainingArguments, DnnCoolSupervisedRunner
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation, synthetic_token_classification, \
    TokenClassificationModel, collate_token_classification


def test_synthetic_dataset():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='security_logs',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    criterion = full_flow_for_development.get_criterion()

    args = TrainingArguments(
        num_epochs=2,
        callbacks=[],
        loaders=nested_loaders,
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
    )

    runner.train(**args)

    print_any_prediction(criterion, model, nested_loaders)


def test_synthetic_dataset_default_runner():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    criterion = full_flow_for_development.get_criterion()
    assert len(criterion.get_metrics()) < 100, 'Callbacks are too many!'

    tensorboard_converters.type_mapping['img'] = [img_publisher]
    tensorboard_converters.type_mapping['text'] = [text_publisher]
    runner.train(num_epochs=10)

    early_stop_callback = runner.default_callbacks[-1]
    assert early_stop_callback.best_score >= 0, 'Negative loss function!'
    print_any_prediction(criterion, model, nested_loaders)


def print_any_prediction(criterion, model, nested_loaders):
    loader = nested_loaders['valid']
    X, y = next(iter(loader))
    X = any2device(X, next(model.parameters()).device)
    y = any2device(y, next(model.parameters()).device)
    model = model.eval()
    pred = model(X)
    res = criterion(pred, y)
    print(res.item())
    print(pred, y)


def test_token_classification_training():
    full_flow_for_development = synthetic_token_classification()
    model = TokenClassificationModel(full_flow_for_development.get_minimal().torch())
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./token_classification',
                                     runner_name='example_run')
    datasets, loaders = runner.get_default_loaders(collator=collate_token_classification)
    runner.train(loaders=loaders, verbose=True)
    print(loaders)

