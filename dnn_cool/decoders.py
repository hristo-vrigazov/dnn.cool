
class Decoder:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def tune(self, predictions, targets):
        raise NotImplementedError()


class BinaryDecoder(Decoder):

    def __init__(self, thresholds=None):
        if thresholds is None:
            print(f'Decoder {self} is not tuned, using default values.')
            thresholds = {'binary': 0.5}
        self.thresholds = thresholds

    def __call__(self, x):
        return x > self.thresholds['binary']

    def tune(self, predictions, targets):
        pass


def threshold_binary(x, threshold=0.5):
    return x > threshold


def sort_declining(x):
    return (-x).argsort(dim=-1)
