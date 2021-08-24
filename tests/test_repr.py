import torch

from dnn_cool.synthetic_dataset import synthetic_dataset_preparation
from dnn_cool.task_flow import BinaryClassificationTask
from torch import nn


def test_repr_binary_task():
    task = BinaryClassificationTask(name='is_car',
                                    labels=torch.ones(24),
                                    module=nn.Linear(256, 1))

    print(task)


def test_repr_flow():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    flow = project.get_synthetic_full_flow()
    print(flow)

