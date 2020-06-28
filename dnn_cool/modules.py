from torch import nn


class SigmoidEval(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training:
            return x
        return self.sigmoid(x)


class SoftmaxEval(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.training:
            return x
        return self.softmax(x)


class SigmoidAndMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        activated_output = self.sigmoid(output)
        return self.mse(activated_output, target)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
