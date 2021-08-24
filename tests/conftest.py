import pytest
import torch
from torch.utils.data import DataLoader

from dnn_cool.runner import DnnCoolSupervisedRunner, DnnCoolRunnerView
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation


@pytest.fixture(scope='package')
def treelib_explanation_on_first_batch():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolRunnerView(
        full_flow=full_flow_for_development.task,
        model=model,
        project_dir='./security_project',
        runner_name='default_experiment',
    )
    model = runner.best()
    n = 4 * torch.cuda.device_count()
    dataset = full_flow_for_development.get_dataset()
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    X, y = next(iter(loader))
    del X['gt']
    model = model.eval()
    res = model(X)
    treelib_explainer = full_flow_for_development.task.get_treelib_explainer()
    tree = treelib_explainer(res)
    return tree
