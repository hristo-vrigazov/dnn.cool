from typing import Dict, Tuple


class IAutoGrad:

    def to_numpy(self, x):
        raise NotImplementedError()

    def as_float(self, x):
        raise NotImplementedError()

    def get_single_float(self, x):
        raise NotImplementedError()


class Tensor:
    def bool(self):
        raise NotImplementedError()


class Dataset:

    def __getitem__(self, item: int) -> Tuple[Dict, Dict]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


def squeeze_if_needed(tensor):
    if len(tensor.shape) > 2:
        return tensor
    if len(tensor.shape) == 2:
        return tensor[:, 0]
    return tensor



