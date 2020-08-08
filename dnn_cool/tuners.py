import torch

from typing import Dict

from dataclasses import dataclass


@dataclass
class TuningData:
    predictions: Dict
    targets: Dict

    def __getattr__(self, item):
        return self


@dataclass
class TunedParameters:
    data: Dict

    def __iadd__(self, other):
        self.data.update(other.data)
        return self

    # Pipeline compatibility
    def __getattr__(self, item):
        return self

    # Pipeline compatibility
    def __invert__(self):
        return self

    # Pipeline compatibility
    def __or__(self, other):
        return self


def get_tuning_data(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, TuningData):
            return arg


class LeafDecoderTuner:

    def __init__(self, task, prefix):
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()
        self.prefix = prefix
        self.path = self.prefix + task.get_name()
        self.available = task.get_available_func()

    def __call__(self, *args, **kwargs):
        tuning_data = get_tuning_data(*args, **kwargs)
        preds = tuning_data.predictions[self.path]
        if self.activation is not None:
            preds = self.activation(torch.tensor(preds).float()).detach().cpu().numpy()
        targets = tuning_data.targets[self.path]

        # TODO: This feels very similar to BaseMetricDecorator. Can we separate the common logic?
        precondition = tuning_data.predictions.get(f'precondition|{self.path}', None)
        if self.available is not None:
            available = self.available(torch.tensor(targets).float()).detach().cpu().numpy()
            if precondition is None:
                precondition = available
            else:
                precondition &= available

        if precondition is None:
            return TunedParameters(self.decoder.tune(preds, targets))
        if precondition.sum() == 0:
            return TunedParameters({self.path: {}})
        if not hasattr(self.decoder, 'tune'):
            raise AttributeError(f'The decoder provided for task {self.path} does not have a "tune" method.')
        tuned_params = self.decoder.tune(preds[precondition], targets[precondition])
        return TunedParameters({self.path: tuned_params})


class CompositeDecoderTuner:

    def __init__(self, task_flow, prefix):
        self.task_flow = task_flow
        self.prefix = prefix

        self.flow = task_flow.get_flow_func()
        self.prefix = prefix
        self.task_name = task_flow.get_name()

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = LeafDecoderTuner(task, prefix)
            else:
                instance = CompositeDecoderTuner(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __call__(self, *args, **kwargs):
        is_root = len(args) == 2
        if is_root:
            predictions, targets = args
        else:
            data: TuningData = args[0]
            predictions = data.predictions
            targets = data.targets

        flow_result = self.flow(self, TuningData(predictions, targets), TunedParameters({}))
        if is_root:
            return flow_result.data
        return flow_result

    def load_tuned(self, tuned_params):
        decoders = self.get_all_decoders()
        for key, decoder in decoders.items():
            decoder.load_tuned(tuned_params[key])

    def get_all_decoders(self):
        decoders = {}
        for task_name, task in self.task_flow.tasks.items():
            if task.has_children():
                decoders.update(getattr(self, task_name).get_all_decoders())
            else:
                decoders[self.prefix + task_name] = getattr(self, task_name).decoder
        return decoders
