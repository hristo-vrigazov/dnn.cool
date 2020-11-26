import torch

from torch.nn import functional as F


class DropoutMC:

    def __init__(self, num_samples=16, p=0.2):
        self.num_samples = num_samples
        self.p = p

    def create_samples(self, module, activation, *args, **kwargs):
        res = []
        for i in range(self.num_samples):
            new_args = [F.dropout(arg, p=self.p, training=True, inplace=False) for arg in args]
            new_kwargs = {key: F.dropout(value, p=self.p, training=True, inplace=False) for key, value in kwargs.items()}
            logits = module(*new_args, **new_kwargs)
            if activation is not None:
                logits = activation(logits)
            res.append(logits)
        return torch.stack(res, dim=0)


