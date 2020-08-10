import torch

from catalyst.utils.metrics import accuracy


def single_result_accuracy(outputs, targets, *args, **kwargs):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(dim=1)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    return accuracy(outputs, targets, *args, **kwargs)[0]
