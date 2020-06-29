import torch
import numpy as np

from dnn_cool.utilities import FlowDict


def test_stores_preconditions():
    res = {
        'door_is_open': {
            'decoded': torch.tensor([True, True, False, True, True]).bool(),
        },
        'person_is_present': {
            'decoded': torch.tensor([True, True, False, False, True]).bool()
        },
        'head_visible': {
            'decoded': torch.tensor([True, True, False, False, False])
        }
    }

    flow_dict = FlowDict(res)

    precondition_mask = FlowDict({
        'decoded': torch.tensor([False, True, True, True, False]).bool()
    })
    result_dict = flow_dict | precondition_mask

    for key, value in result_dict.preconditions.items():
        assert np.allclose(precondition_mask.decoded, value)


def test_chaining_preconditions_work_correctly():
    res = {
        'door_is_open': {
            'decoded': torch.tensor([True, True, False, True, True]).bool(),
        },
        'person_is_present': {
            'decoded': torch.tensor([True, True, False, False, True]).bool()
        },
        'head_visible': {
            'decoded': torch.tensor([True, True, False, False, False])
        }
    }

    flow_dict = FlowDict(res)

    first_precondition = torch.tensor([False, True, True, True, False]).bool()
    second_precondition = torch.tensor([True, True, False, True, False]).bool()
    third_precondition = torch.tensor([True, True, True, False, True]).bool()
    expected = first_precondition & second_precondition & third_precondition

    for precondition in [first_precondition, second_precondition, third_precondition]:
        precondition_mask = FlowDict({
            'decoded': precondition
        })
        flow_dict |= precondition_mask

    for key, actual in flow_dict.preconditions.items():
        assert np.allclose(expected, actual)
