import torch

from typing import Dict


class FlowDict:

    def __init__(self, res: Dict):
        self.res = res
        self.preconditions = {}

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

    def __xor__(self, other):
        res = {
            'decoded': self.decoded ^ other.decoded
        }
        return self.__shallow_copy_keys(res)

    def __sub__(self, other):
        res = FlowDict({})
        for key, value in self.res.items():
            if key not in other.res:
                res.res[key] = value
        for key, value in self.preconditions.items():
            if key not in other.preconditions:
                res.preconditions[key] = value
        return res

    def __add__(self, other):
        res = FlowDict({})
        res.res.update(self.res)
        res.res.update(other.res)
        res.preconditions.update(self.preconditions)
        res.preconditions.update(other.preconditions)
        return res

    def __iadd__(self, other):
        self.res.update(other.res)
        self.preconditions.update(other.preconditions)
        return self

    def flatten(self):
        res = {}
        for key, value in self.preconditions.items():
            res[f'precondition|{key}'] = value

        for key, value in self.res.items():
            path_key, path_value = self._traverse(key, value)
            res[path_key] = path_value

        return res

    def _traverse(self, path_so_far, subtree):
        try:
            return path_so_far, subtree.logits
        except:
            raise NotImplementedError(f'Converting for nested FlowDicts is not implemented yet.')