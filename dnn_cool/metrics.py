import torch

from catalyst.utils.metrics import accuracy


class Metric:

    def __init__(self, metric_fc, is_multimetric=False, list_args=None):
        self.activation = None
        self.decoder = None
        self.metric_fc = metric_fc
        self._is_multimetric = is_multimetric
        self._list_args = list_args

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
        return self._invoke_metric(outputs, targets)

    def _invoke_metric(self, outputs, targets):
        return self.metric_fc(outputs, targets)

    def is_multi_metric(self):
        return self._is_multimetric

    def list_args(self):
        return self._list_args


class Accuracy(Metric):

    def __init__(self):
        super().__init__(accuracy, is_multimetric=True, list_args=(1, 3, 5))


class NumpyMetric(Metric):

    def __init__(self, metric_fc):
        super().__init__(metric_fc)

    def _invoke_metric(self, outputs, targets, activate=True, decode=True):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        return self.metric_fc(outputs, targets)


def single_result_accuracy(outputs, targets, *args, **kwargs):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(dim=1)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    return accuracy(outputs, targets, *args, **kwargs)[0]
