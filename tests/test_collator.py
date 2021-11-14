import torch

from dnn_cool.collators.base import find_padding_shape_of_nested_list, examples_to_nested_list


def test_finds_max_along_every_axis():
    ll = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5])
    ]
    res = find_padding_shape_of_nested_list(ll)
    assert res == [2, 3]


def test_finds_max_along_every_axis_3_axes():
    ll = [
        [
            torch.tensor([1, 2, 3]),
        ], [
            torch.tensor([4]),
            torch.tensor([6])
        ], [
            torch.tensor([9])
        ], [
            torch.tensor([12])
        ]
    ]
    res = find_padding_shape_of_nested_list(ll)
    assert res == [4, 2, 3]


def test_finds_max_along_every_axis_5_axes():
    ll = create_nested_list_test()
    res = find_padding_shape_of_nested_list(ll)
    assert res == [3, 2, 6, 1, 5]


def create_nested_list_test():
    ll000 = [torch.tensor([1, 2, 3, 4, 5])]
    ll010 = [torch.tensor([1])]
    ll00 = [ll000, ll000, ll000, ll000, ll000, ll000]
    ll01 = [ll010]
    ll0 = [ll00, ll01]
    ll = [ll0, ll0, ll0]
    return ll


def test_examples_to_nested_list():
    examples = [
        ({'task_name': create_nested_list_test()}, {'label': torch.tensor([5])})
    ]
    r = examples_to_nested_list(examples)
    print(r)
