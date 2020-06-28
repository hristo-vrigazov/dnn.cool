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


# TODO: is there a way to vectorize this in case all counts are different?
class NestedFC(nn.Module):

    def __init__(self, in_features, out_features_nested, bias):
        super().__init__()
        fcs = []
        for i, out_features in enumerate(out_features_nested):
            fcs.append(nn.Linear(in_features, out_features, bias))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, features, parent_indices):
        """
        features and parent_indices must have the same length. Iterates over parent indices, and records predictions
        for every pair (N, P), where N is a batch number and P is a parent index. returns list of lists.
        :param features:
        :param parent_indices:
        :return: list of lists of lists, where every element is the prediction of the respective module. The first
        len is equal to the batch size, the second len is equal to the top_k and the third len is equal to the respective
        number of classes for the child FC.
        """
        raise NotImplementedError()