import torch
import pytest
import numpy as np

from dnn_cool.utilities import FlowDict


@pytest.fixture()
def example_data():
    res = {
        'door_is_open': {
            'decoded': torch.tensor([True, True, False, True, True]).bool(),
        },
        'person_is_present': {
            'decoded': torch.tensor([True, True, False, False, True]).bool()
        },
        'head_visible': {
            'decoded': torch.tensor([True, True, False, False, False]).bool()
        }
    }
    flow_dict = FlowDict(res)
    first_precondition = torch.tensor([False, True, True, True, False]).bool()
    second_precondition = torch.tensor([True, True, False, True, False]).bool()
    third_precondition = torch.tensor([True, True, True, False, True]).bool()
    precondition_dicts = []
    for precondition in [first_precondition, second_precondition, third_precondition]:
        precondition_mask = FlowDict({
            'decoded': precondition
        })
        precondition_dicts.append(precondition_mask)
    return flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition


def test_stores_preconditions(example_data):
    flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition = example_data
    expected = first_precondition

    for precondition in [first_precondition]:
        precondition_mask = FlowDict({
            'decoded': precondition
        })
        flow_dict |= precondition_mask

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)


def test_chaining_preconditions_work_correctly(example_data):
    flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition = example_data
    expected = first_precondition & second_precondition & third_precondition

    for precondition in [first_precondition, second_precondition, third_precondition]:
        precondition_mask = FlowDict({
            'decoded': precondition
        })
        flow_dict |= precondition_mask

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)


def test_mask_operations(example_data):
    flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition = example_data

    expected = first_precondition & second_precondition & (~third_precondition)
    flow_dict = flow_dict | (precondition_dicts[0] & precondition_dicts[1] & (~precondition_dicts[2]))

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)


def test_mask_operations_or(example_data):
    flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition = example_data

    expected = first_precondition | second_precondition
    flow_dict = flow_dict | (precondition_dicts[0].or_else(precondition_dicts[1]))

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)


def test_mask_operations_xor(example_data):
    flow_dict, precondition_dicts, first_precondition, second_precondition, third_precondition = example_data

    expected = first_precondition ^ second_precondition
    flow_dict = flow_dict | (precondition_dicts[0] ^ (precondition_dicts[1]))

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)

