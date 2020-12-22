from functools import partial

import torch

from catalyst.utils.metrics import accuracy, multilabel_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn


class TorchMetric:

    def __init__(self, metric_fn, decode=True, metric_args=None):
        if metric_args is None:
            metric_args = {}
        self.activation = None
        self.decoder = None
        self.metric_fn = metric_fn
        self.metric_args = metric_args
        self._decode = decode
        self._is_binded = False

    def bind_to_task(self, task):
        self._is_binded = True
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()

    def __call__(self, outputs, targets, activate=True):
        if not self._is_binded:
            raise ValueError(f'The metric is not binded to a task, but is already used.')
        outputs = torch.as_tensor(outputs)
        targets = torch.as_tensor(targets)

        if activate:
            outputs = self.activation(outputs)
        if self._decode:
            outputs = self.decoder(outputs)
        return self._invoke_metric(outputs, targets, self.metric_args)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        return self.metric_fn(outputs, targets, **metric_args_dict)


class BinaryAccuracy(TorchMetric):

    def __init__(self):
        super().__init__(accuracy, decode=True)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        if len(outputs.shape) <= 1:
            outputs = outputs.unsqueeze(dim=-1)
        return self.metric_fn(outputs, targets, **metric_args_dict)[0]


class ClassificationAccuracy(TorchMetric):

    def __init__(self, metric_args=None):
        if metric_args is None:
            metric_args = {'topk': [1]}
        super().__init__(accuracy, decode=False, metric_args=metric_args)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        topk = metric_args_dict['topk']
        results = self.metric_fn(outputs, targets, **metric_args_dict)
        dict_metrics = dict(zip(topk, results))
        return dict_metrics

    def empty_precondition_result(self):
        res = {}
        for metric_arg in self.metric_args['topk']:
            res[metric_arg] = torch.tensor(0.)
        return res


class NumpyMetric(TorchMetric):

    def __init__(self, metric_fn, decode=True):
        super().__init__(metric_fn, decode)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        return self.metric_fn(outputs, targets, **metric_args_dict)


class BinaryF1Score(NumpyMetric):

    def __init__(self):
        super().__init__(f1_score)


class BinaryPrecision(NumpyMetric):

    def __init__(self):
        super().__init__(precision_score)


class BinaryRecall(NumpyMetric):

    def __init__(self):
        super().__init__(recall_score)


class ClassificationNumpyMetric(NumpyMetric):

    def __init__(self, metric_fn, decode=True):
        super().__init__(metric_fn, decode=decode)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        outputs = outputs[..., 0]
        return self.metric_fn(outputs, targets, **metric_args_dict)


class ClassificationF1Score(ClassificationNumpyMetric):

    def __init__(self):
        super().__init__(partial(f1_score, average='micro'))


class ClassificationPrecision(ClassificationNumpyMetric):

    def __init__(self):
        super().__init__(partial(precision_score, average='micro'))


class ClassificationRecall(ClassificationNumpyMetric):

    def __init__(self):
        super().__init__(partial(recall_score, average='micro'))


class MeanAbsoluteError(TorchMetric):

    def __init__(self, decode=False):
        super().__init__(nn.L1Loss(), decode)


class MultiLabelClassificationAccuracy(TorchMetric):

    def __init__(self, metric_args=None):
        if metric_args is None:
            metric_args = {'threshold': 0.5}
        super().__init__(multilabel_accuracy, decode=True, metric_args=metric_args)

    def _invoke_metric(self, outputs, targets, metric_args_dict):
        # the threshold does not actually matter, since the outputs are already decoded, i.e they are already 1 and 0
        return self.metric_fn(outputs, targets, **metric_args_dict)


def get_default_binary_metrics():
    return (
             ('accuracy', BinaryAccuracy()),
             ('f1_score', BinaryF1Score()),
             ('precision', BinaryPrecision()),
             ('recall', BinaryRecall()),
    )


def get_default_bounded_regression_metrics():
    return (
        ('mean_absolute_error', MeanAbsoluteError()),
    )


def get_default_classification_metrics():
    return (
        ('accuracy', ClassificationAccuracy()),
        ('f1_score', ClassificationF1Score()),
        ('precision', ClassificationPrecision()),
        ('recall', ClassificationRecall()),
    )


def get_default_multilabel_classification_metrics():
    return (
        ('accuracy', MultiLabelClassificationAccuracy()),
    )
