from dnn_cool.tuners import CompositeDecoderTuner


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
        return {}


class DecoderDecorator:

    def __init__(self, decoder, prefix):
        self.decoder = decoder
        self.prefix = prefix


class CompositeDecoder:

    def __init__(self, task_flow, prefix):
        self.prefix = prefix
        self.task_flow = task_flow

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TaskFlowDecoder(Decoder):

    def __init__(self, task_flow, prefix=''):
        self.tuner = CompositeDecoderTuner(task_flow, prefix)
        self.composite_decoder = CompositeDecoder(task_flow, prefix)

    def __call__(self, *args, **kwargs):
        return self.composite_decoder(*args, **kwargs)

    def tune(self, predictions, targets):
        return self.tuner(predictions, targets)


def threshold_binary(x, threshold=0.5):
    return x > threshold


def sort_declining(x):
    return (-x).argsort(dim=-1)
