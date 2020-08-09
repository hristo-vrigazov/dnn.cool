from dataclasses import dataclass, field
from typing import Dict

from dnn_cool.synthetic_dataset import synthenic_dataset_preparation
from dnn_cool.visitors import RootCompositeVisitor, VisitorOut, LeafVisitor


def test_simple_visitor():
    model, nested_loaders, datasets, project = synthenic_dataset_preparation()
    runner = project.runner(model=model, runner_name='security_logs')
    flow = project.get_full_flow()

    class TuningVisitor(LeafVisitor):

        def __init__(self, task, prefix):
            super().__init__(task, prefix)

        def full_result(self, preds, targets):
            return TunedParams(self.decoder.tune(preds, targets))

        def empty_result(self):
            return TunedParams({self.path: {}})

        def preconditioned_result(self, preds, targets):
            tuned_params = self.decoder.tune(preds, targets)
            return TunedParams({self.path: tuned_params})

    @dataclass
    class TunedParams(VisitorOut):
        data: Dict = field(default_factory=lambda: {})

        def __iadd__(self, other):
            self.data.update(other.data)
            return self

    visitor = RootCompositeVisitor(flow, TuningVisitor, TunedParams)
    predictions, targets, intepretations = runner.load_inference_results()
    res = visitor(predictions['valid'], targets['valid'])
    print(res)
