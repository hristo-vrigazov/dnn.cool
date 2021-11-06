from dataclasses import dataclass, field
from typing import Dict

from dnn_cool.tuners import TunerVisitor
from dnn_cool.visitors import LeafVisitor, VisitorOut, RootCompositeVisitor


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


class DecodingVisitor(LeafVisitor):

    def __init__(self, task, prefix, autograd):
        super().__init__(task, prefix, autograd)

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

    def __init__(self, task_flow, prefix, autograd):
        super().__init__(task_flow, DecodingVisitor, DecodedData, prefix, autograd)


class TaskFlowDecoder(Decoder):

    def __init__(self, task_flow, prefix, autograd):
        self.tuner = TunerVisitor(task_flow, prefix=prefix, autograd=autograd)
        self.composite_decoder = CompositeDecoder(task_flow, prefix, autograd)

    def __call__(self, *args, **kwargs):
        return self.composite_decoder(*args, **kwargs)

    def tune(self, predictions, targets):
        return self.tuner(predictions, targets)

    def load_tuned(self, tuned_params):
        self.tuner.load_tuned(tuned_params)


class NoOpDecoder(Decoder):
    def __call__(self, x):
        return x

    def tune(self, predictions, targets):
        pass

    def load_tuned(self, params):
        pass
