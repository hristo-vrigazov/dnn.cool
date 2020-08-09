import torch

from typing import Dict

from dataclasses import dataclass, field

from dnn_cool.synthetic_dataset import synthenic_dataset_preparation
from dnn_cool.visitors import RootCompositeVisitor, VisitorOut, VisitorData


def test_simple_visitor():
    model, nested_loaders, datasets, project = synthenic_dataset_preparation()
    runner = project.runner(model=model, runner_name='security_logs')
    flow = project.get_full_flow()

    class TuningData(VisitorData):

        def visit(self, task, prefix):
            path = prefix + task.get_name()
            preds = self.predictions[path]
            activation = task.get_activation()
            if activation is not None:
                preds = activation(torch.tensor(preds).float()).detach().cpu().numpy()
            targets = self.targets[path]

            precondition = self.predictions.get(f'precondition|{self.path}', None)
            available_func = task.get_available_func()
            if available_func is not None:
                available = available_func(torch.tensor(targets).float()).detach().cpu().numpy()
                if precondition is None:
                    precondition = available
                else:
                    precondition &= available

            decoder = task.get_decoder()
            if precondition is None:
                return TunedParams({path: decoder.tune(preds, targets)})
            if precondition.sum() == 0:
                return TunedParams({path: {}})
            if not hasattr(self.decoder, 'tune'):
                raise AttributeError(f'The decoder provided for task {self.path} does not have a "tune" method.')
            tuned_params = decoder.tune(preds[precondition], targets[precondition])
            return TunedParams({path: tuned_params})

    @dataclass
    class TunedParams(VisitorOut):
        data: Dict = field(default_factory=lambda: {})

        def __iadd__(self, other):
            self.data.update(other.data)
            return self

    visitor = RootCompositeVisitor(flow, TuningData, TunedParams)
    predictions, targets, intepretations = runner.load_inference_results()
    res = visitor(predictions['valid'], targets['valid'])
    print(res)
