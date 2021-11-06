from dnn_cool.decoders.base import Decoder


def sort_declining(x):
    return (-x).argsort(-1)


class ClassificationDecoder(Decoder):

    def __call__(self, x):
        return sort_declining(x)

    def tune(self, predictions, targets):
        return {}

    def load_tuned(self, params):
        pass
