import torch
import pytest

from dnn_cool.datasets import FlowDataset
from dnn_cool.converters import Values
from dnn_cool.task_flow import TaskFlow, BinaryHardcodedTask


@pytest.fixture()
def example_numerical_flow():
    def numerical_flow(flow, x, out):
        out += flow.is_even(x.features)
        out += flow.predict_positive(x.features) | out.is_even
        out += flow.multiple_three(x.features) | out.predict_positive
        return out

    def full_flow(flow, x, out):
        out += flow.is_interesting(x.features)
        out += flow.numerical_flow(x.features) | out.is_interesting
        return out

    tensor = torch.arange(8).unsqueeze(dim=-1)
    inputs = Values(keys=['inp'], values=[tensor])

    is_even_task = BinaryHardcodedTask(name='is_even', labels=(tensor % 2) == 0)
    predict_positive = BinaryHardcodedTask(name='predict_positive', labels=(tensor > 0.))
    multiple_three = BinaryHardcodedTask(name='multiple_three', labels=(tensor % 3) == 0)
    is_interesting_task = BinaryHardcodedTask(name='is_interesting', labels=torch.tensor(
        [True, True, False, False, False, True, True, True]))

    tasks = [is_even_task, predict_positive, multiple_three]

    numerical_flow_task = TaskFlow(name='numerical_flow', tasks=tasks, inputs=inputs, flow_func=numerical_flow)
    full_task_flow = TaskFlow(name='full_flow', tasks=[is_interesting_task, numerical_flow_task], inputs=inputs, flow_func=full_flow)
    return full_task_flow


def test_includes_everything_needed_and_stores_gt(example_numerical_flow):
    dataset = FlowDataset(example_numerical_flow)

    X, y = dataset[0]

    assert 'is_interesting' in X['gt']
    assert 'numerical_flow.is_even' in X['gt']
    assert 'numerical_flow.predict_positive' in X['gt']
    assert not 'numerical_flow.multiple_three' in X['gt']
