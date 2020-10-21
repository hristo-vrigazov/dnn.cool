import torch


def positive_values(tensor):
    mask = tensor >= 0.
    axes = tuple(range(1, len(mask.shape)))
    if len(axes) > 0:
        mask = mask.sum(axis=axes) > 0
    return mask


def all_correct(tensor):
    return torch.ones(len(tensor), dtype=bool)
