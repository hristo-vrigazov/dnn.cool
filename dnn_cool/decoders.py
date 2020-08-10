from typing import Dict

import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from dnn_cool.tuners import TuningVisitor, TunedParams, TunerVisitor
from dnn_cool.visitors import RootCompositeVisitor, VisitorOut, LeafVisitor


class Decoder:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def tune(self, predictions, targets):
        raise NotImplementedError()

    def load_tuned(self, params):
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
        res = np.zeros_like(self._candidates)
        for i, candidate in enumerate(tqdm(self._candidates)):
            preds = (predictions > candidate)
            res[i] = self.metric(preds, targets)
        params = {'threshold': self._candidates[res.argmax()]}
        self.load_tuned(params)
        return params

    def load_tuned(self, params):
        self.threshold = params['threshold']


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


def threshold_binary(x, threshold=0.5):
    return x > threshold


def sort_declining(x):
    return (-x).argsort(dim=-1)
