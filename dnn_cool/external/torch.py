import torch

from dnn_cool.external.autograd import IAutoGrad


class TorchAutoGrad(IAutoGrad):

    def as_float(self, x):
        return torch.as_tensor(x).float()

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def get_single_float(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return x

