from dataclasses import dataclass
from typing import Dict


def initialize_dict_structure(dct):
    res = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            res[key] = initialize_dict_structure(value)
            continue
        res[key] = []
    return res


def find_max_len_of_list_of_lists(ll):
    r = [max(len(l) for l in ll)]
    tmp = ll
    child = tmp[0]
    while isinstance(child, list):
        r.append(max(len(l) for l in child))
        tmp = child
        child = tmp[0]
    return r


def find_padding_shape_of_nested_list(ll):
    return [len(ll)] + find_max_len_of_list_of_lists(ll)


def append_example_to_dict(dct, ex):
    for key, value in ex.items():
        if isinstance(value, dict):
            dct[key] = append_example_to_dict(dct[key], value)
            continue
        dct[key].append(value)
    return dct


def create_shapes_dict(dct):
    res = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            res[key] = create_shapes_dict(value)
            continue
        res[key] = find_padding_shape_of_nested_list(value)
    return res


def apply_to_nested_dict(dct, func):
    res = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            res[key] = apply_to_nested_dict(value, func)
            continue
        res[key] = func(key, value)
    return res


@dataclass
class CollatorData:
    X_batch: Dict
    y_batch: Dict
    X_shapes: Dict
    y_shapes: Dict


def examples_to_nested_list(examples):
    X_ex, y_ex = examples[0]
    X_batch = initialize_dict_structure(X_ex)
    y_batch = initialize_dict_structure(y_ex)
    for example in examples:
        X_ex, y_ex = example
        append_example_to_dict(X_batch, X_ex)
        append_example_to_dict(y_batch, y_ex)

    return CollatorData(
        X_batch=X_batch,
        y_batch=y_batch,
        X_shapes=create_shapes_dict(X_batch),
        y_shapes=create_shapes_dict(y_batch)
    )
