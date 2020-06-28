import torch

from typing import Dict


class FlowDict:

    def __init__(self, res: Dict):
        self.res = res

    def __iadd__(self, other):
        self.res.update(other.res)
        return self

    def __getattr__(self, attr):
        return self.res[attr]

    def __or__(self, result):
        mask_dict = {}
        for key, value in self.res.items():
            mask_dict[f'precondition|{key}'] = result
        self.res.update(mask_dict)
        return self

    def __invert__(self):
        for key, value in self.res.items():
            self.res[key] = ~value
        return self
