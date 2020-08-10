from typing import Dict

from dataclasses import dataclass, field

from dnn_cool.visitors import LeafVisitor, VisitorOut, RootCompositeVisitor


class FilterVisitor(LeafVisitor):

    def __init__(self, task, prefix):
        super().__init__(task, prefix)

    def full_result(self, preds, targets):
        return FilterParams(predictions={self.path: preds}, targets={self.path: targets})

    def empty_result(self):
        return FilterParams(predictions={}, targets={})

    def preconditioned_result(self, preds, targets):
        return FilterParams(predictions={self.path: preds}, targets={self.path: targets})


@dataclass
class FilterParams(VisitorOut):
    predictions: Dict = field(default_factory=lambda: {})
    targets: Dict = field(default_factory=lambda: {})

    def __iadd__(self, other):
        self.predictions.update(other.predictions)
        self.targets.update(other.targets)
        return self

    def reduce(self):
        return {
            'predictions': self.predictions,
            'targets': self.targets
        }


class FilterCompositeVisitor(RootCompositeVisitor):

    def __init__(self, task_flow, prefix):
        super().__init__(task_flow, FilterVisitor, FilterParams, prefix=prefix)

    def load_tuned(self, tuned_params):
        tasks = self.task_flow.get_all_children()
        for path, task in tasks.items():
            task.get_decoder().load_tuned(tuned_params[path])
