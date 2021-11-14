from dataclasses import dataclass
from typing import Dict

from dnn_cool.dsl import IFeaturesDict


def find_arg_with_gt(args, is_kwargs):
    new_args = {} if is_kwargs else []
    gt = None
    for arg in args:
        if is_kwargs:
            arg = args[arg]
        try:
            gt = arg['gt']
            new_args.append(arg['value'])
        except:
            new_args.append(arg)
    return gt, new_args


def select_gt(a_gt, k_gt):
    if a_gt is None and k_gt is None:
        return {}

    if a_gt is None:
        return k_gt

    return a_gt


def find_gt_and_process_args_when_training(*args, **kwargs):
    a_gt, args = find_arg_with_gt(args, is_kwargs=False)
    k_gt, kwargs = find_arg_with_gt(kwargs, is_kwargs=True)
    return args, kwargs, select_gt(a_gt, k_gt)


def _copy_to_self(self_arr, other_arr):
    for key, value in self_arr.items():
        assert key not in other_arr, f'The key {key} has been added twice in the same workflow!.'
        other_arr[key] = value


@dataclass
class FeaturesDict(IFeaturesDict):
    data: Dict

    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):
        return self.data[item]