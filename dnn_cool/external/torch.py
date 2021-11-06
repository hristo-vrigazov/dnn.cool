import torch

from dnn_cool.external.autograd import IAutoGrad


class TorchAutoGrad(IAutoGrad):

    def as_float(self, x):
        return torch.tensor(x).float()

    def to_numpy(self, x):
        return x.detach().cpu().numpy()
