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
        """
        Note that this has the meaning of preconditioning, not boolean __or__. If you want to use boolean or,
        use the `or_else` method.
        :param result:
        :return:
        """
        for key in self.res:
            current_precondition = self.preconditions.get(key, result.decoded)
            self.preconditions[key] = current_precondition & result.decoded
        return self

    def __invert__(self):
        res = {
            'decoded': ~self.decoded
        }
        return self.__shallow_copy_keys(res)

    def __and__(self, other):
        res = {
            'decoded': self.decoded & other.decoded
        }
        return self.__shallow_copy_keys(res)

    def __getitem__(self, key):
        return self.res[key]

    def __setitem__(self, key, item):
        self.res[key] = item

    def __iter__(self):
        return self.res.__iter__()

    def __contains__(self, item):
        return self.res.__contains__(item)

    def __shallow_copy_keys(self, res):
        for key, value in self.res.items():
            if key not in res:
                res[key] = value
        return FlowDict(res)

    def or_else(self, other):
        res = {
            'decoded': self.decoded | other.decoded
        }
        return self.__shallow_copy_keys(res)