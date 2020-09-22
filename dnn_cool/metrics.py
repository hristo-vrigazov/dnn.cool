from functools import partial

import torch

from catalyst.utils.metrics import accuracy, multi_label_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn


class TorchMetric:

    def __init__(self, metric_fn, decode=True):
        self.activation = None
        self.decoder = None
        self.metric_fn = metric_fn
        self._decode = decode

    def bind_to_task(self, task):
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()

    def __call__(self, outputs, targets, activate=True):
        if (self.activation is None) or (self.decoder is None):
            raise ValueError(f'The metric is not binded to a task, but is already used.')
        outputs = torch.as_tensor(outputs)
        targets = torch.as_tensor(targets)

        if activate:
            outputs = self.activation(outputs)
        if self._decode:
            outputs = self.decoder(outputs)
        return self._invoke_metric(outputs, targets)

    def _invoke_metric(self, outputs, targets):
        return self.metric_fn(outputs, targets)


class BinaryAccuracy(TorchMetric):

    def __init__(self):
        super().__init__(accuracy, decode=True)

    def _invoke_metric(self, outputs, targets):
        if len(outputs.shape) <= 1:
            outputs = outputs.unsqueeze(dim=-1)
        return self.metric_fn(outputs, targets)[0]


class ClassificationAccuracy(TorchMetric):

    def __init__(self):
        super().__init__(accuracy, decode=False)

    def _invoke_metric(self, outputs, targets):
        n_classes = outputs.shape[-1]
        topk = [1]
        if n_classes > 3:
            topk.append(3)
        if n_classes > 5:
            topk.append(5)
        results = self.metric_fn(outputs, targets, topk=topk)
        dict_metrics = dict(zip(topk, results))
        return dict_metrics


class NumpyMetric(TorchMetric):

    def __init__(self, metric_fn, decode=True):
        super().__init__(metric_fn, decode)

    def _invoke_metric(self, outputs, targets):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        return self.metric_fn(outputs, targets)


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

    def _invoke_metric(self, outputs, targets):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        outputs = outputs[..., 0]
        return self.metric_fn(outputs, targets)


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

    def __init__(self):
        super().__init__(multi_label_accuracy, decode=True)

    def _invoke_metric(self, outputs, targets):
        # the threshold does not actually matter, since the outputs are already decoded, i.e they are already 1 and 0
        return self.metric_fn(outputs, targets, threshold=0.5)


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
