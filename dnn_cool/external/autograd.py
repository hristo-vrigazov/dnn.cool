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


def squeeze_last_dim_if_needed(tensor):
    if tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor



