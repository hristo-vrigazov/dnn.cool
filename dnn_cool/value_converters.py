import numpy as np
import torch


def binary_value_converter(values):
    values[np.isnan(values.astype(float))] = -1
    return torch.tensor(values.astype(float)).float().unsqueeze(dim=-1)

