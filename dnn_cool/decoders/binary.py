import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from dnn_cool.decoders.base import Decoder


class BinaryDecoder(Decoder):

    def __init__(self, threshold=None, metric=f1_score, start=0.01, end=0.99):
        if threshold is None:
            threshold = 0.5
        self.threshold = threshold

        self._candidates = np.linspace(start, end, num=100)
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