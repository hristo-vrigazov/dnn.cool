from dnn_cool.decoders.base import Decoder


class BoundedRegressionDecoder(Decoder):

    def __init__(self, scale=224):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale

    def tune(self, predictions, targets):
        return {}

    def load_tuned(self, params):
        pass
