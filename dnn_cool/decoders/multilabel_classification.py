import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dnn_cool.decoders.base import Decoder


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