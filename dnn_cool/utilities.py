import torch

from typing import Dict


class FlowDict:

    def __init__(self, res: Dict):
        self.res = res
        self.preconditions = {}

    def __iadd__(self, other):
        self.res.update(other.res)
        self.preconditions.update(other.preconditions)
        return self

    def __getattr__(self, attr):
        if not ('training' in self.res):
            return self.res[attr]
        if not self.res['training']:
            return self.res[attr]
        return {
            'value': self.res[attr],
            'gt': self.res['gt'],
        }

    def __or__(self, result):
        for key in self.res:
            self.preconditions[key] = result.decoded
        return self

    def __invert__(self):
        res = {
            'decoded': ~self.decoded
        }
        for key, value in self.res.items():
            res[key] = value
        return FlowDict(res)

    def __and__(self, other):
        res = {
            'decoded': self.decoded & other.decoded
        }
        for key, value in self.res.items():
            res[key] = value
        return FlowDict(res)
