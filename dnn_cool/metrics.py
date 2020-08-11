import torch

from catalyst.utils.metrics import accuracy


class Metric:

    def __init__(self, metric_fc):
        self.activation = None
        self.decoder = None
        self.metric_fc = metric_fc

    def bind_to_task(self, task):
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()

    def __call__(self, outputs, targets, activate=True, decode=True):
        if (self.activation is None) or (self.decoder is None):
            raise ValueError(f'The metric is not binded to a task, but is already used.')

        if activate:
            outputs = self.activation(outputs)
        if decode:
            outputs = self.decoder(outputs)
        return self.metric_fc(outputs, targets)


class Accuracy(Metric):

    def __init__(self):
        super().__init__(single_result_accuracy)


def single_result_accuracy(outputs, targets, *args, **kwargs):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(dim=1)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    return accuracy(outputs, targets, *args, **kwargs)[0]
