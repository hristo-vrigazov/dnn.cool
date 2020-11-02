import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Dict

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dnn_cool.tuners import TunerVisitor
from dnn_cool.visitors import RootCompositeVisitor, VisitorOut, LeafVisitor


class Decoder:
    """
    A decoder is a :class:`Callable` which when invoked, returns processed task predictions so that they can be used
    as a precondition to other tasks.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def tune(self, predictions, targets):
        """
        Tunes any thresholds or other parameters needed for decoding.

        :param predictions: The activated model predictions

        :param targets: The ground truth

        :return: A dictionary of tuned parameters.
        """
        raise NotImplementedError()

    def load_tuned(self, params):
        """
        Loads a dictionary of tuned parameters.

        :param params: a dictionary of tuned parameters.

        """
        raise NotImplementedError()


class BinaryDecoder(Decoder):

    def __init__(self, threshold=None, metric=accuracy_score):
        if threshold is None:
            print(f'Decoder {self} is not tuned, using default values.')
            threshold = 0.5
        self.threshold = threshold

        self._candidates = np.linspace(0., 1., num=100)
        self.metric = metric

    def __call__(self, x):
        return x > self.threshold

    def tune(self, predictions, targets):
        """
        Performs a simple grid search that optimizes the metric in the field :code:`metric`.

        :param predictions: The activated model predictions

        :param targets: The ground truth

        :return: A dictionary of tuned parameters.
        """
        res = np.zeros_like(self._candidates)
        for i, candidate in enumerate(tqdm(self._candidates)):
            preds = (predictions > candidate)
            res[i] = self.metric(preds, targets)
        params = {'threshold': self._candidates[res.argmax()]}
        self.load_tuned(params)
        return params

    def load_tuned(self, params):
        self.threshold = params['threshold']


class ClassificationDecoder(Decoder):

    def __call__(self, x):
        return sort_declining(x)

    def tune(self, predictions, targets):
        return {}

    def load_tuned(self, params):
        pass


class DecodingVisitor(LeafVisitor):

    def __init__(self, task, prefix):
        super().__init__(task, prefix)

    def full_result(self, preds, targets):
        return DecodedData({self.path: self.decoder(preds)})

    def empty_result(self):
        return DecodedData({self.path: {}})

    def preconditioned_result(self, preds, targets):
        return DecodedData({self.path: self.decoder(preds)})


@dataclass
class DecodedData(VisitorOut):
    data: Dict = field(default_factory=lambda: {})

    def __iadd__(self, other):
        self.data.update(other.data)
        return self

    def reduce(self):
        return self.data


class CompositeDecoder(RootCompositeVisitor):

    def __init__(self, task_flow, prefix):
        super().__init__(task_flow, DecodingVisitor, DecodedData, prefix)


class TaskFlowDecoder(Decoder):

    def __init__(self, task_flow, prefix=''):
        self.tuner = TunerVisitor(task_flow, prefix=prefix)
        self.composite_decoder = CompositeDecoder(task_flow, prefix)

    def __call__(self, *args, **kwargs):
        return self.composite_decoder(*args, **kwargs)

    def tune(self, predictions, targets):
        return self.tuner(predictions, targets)

    def load_tuned(self, tuned_params):
        self.tuner.load_tuned(tuned_params)


class BoundedRegressionDecoder(Decoder):

    def __init__(self, scale=224):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale

    def tune(self, predictions, targets):
        return {}

    def load_tuned(self, params):
        pass


def threshold_binary(x, threshold=0.5):
    return x > threshold


def sort_declining(x):
    return (-x).argsort(-1)


class MultilabelClassificationDecoder(Decoder):

    def __init__(self, metric=accuracy_score):
        self.thresholds = None
        self._candidates = np.linspace(0., 1., num=100)
        self.metric = metric

    def __call__(self, x):
        if self.thresholds is None:
            n_classes = x.shape[-1]
            self.thresholds = torch.ones(n_classes).unsqueeze(0) * 0.5
        if isinstance(x, np.ndarray):
            return x > self.thresholds.cpu().numpy()
        return x > self.thresholds.to(x.device)

    def tune(self, predictions, targets):
        n_classes = predictions.shape[-1]
        res = np.zeros((n_classes, len(self._candidates)), dtype=float)
        for class_idx in range(n_classes):
            for j, candidate in enumerate(tqdm(self._candidates)):
                preds = (predictions[:, class_idx] > candidate)
                gt = targets[:, class_idx]
                res[class_idx, j] = self.metric(preds, gt)

        best_threshold_indices = res.argmax(axis=1)
        params = {'thresholds': self._candidates[best_threshold_indices]}
        self.load_tuned(params)
        return params

    def load_tuned(self, params):
        self.thresholds = torch.tensor(params['thresholds']).unsqueeze(0)


class NoOpDecoder(Decoder):
    def __call__(self, x):
        return x

    def tune(self, predictions, targets):
        pass

    def load_tuned(self, params):
        pass