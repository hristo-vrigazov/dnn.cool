from dnn_cool.tuners import CompositeDecoderTuner


class Decoder:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def tune(self, predictions, targets):
        raise NotImplementedError()

    def load_tuned(self, params):
        raise NotImplementedError()


class BinaryDecoder(Decoder):

    def __init__(self, threshold=None):
        if threshold is None:
            print(f'Decoder {self} is not tuned, using default values.')
            threshold = {'binary': 0.5}
        self.threshold = threshold

    def __call__(self, x):
        return x > self.threshold

    def tune(self, predictions, targets):
        return {'threshold': 0.21}

    def load_tuned(self, params):
        self.threshold = params['threshold']


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
