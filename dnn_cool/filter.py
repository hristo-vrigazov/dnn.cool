from typing import Dict

from dataclasses import dataclass, field

from dnn_cool.visitors import LeafVisitor, VisitorOut, RootCompositeVisitor


class FilterVisitor(LeafVisitor):

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
class FilterParams(VisitorOut):
    data: Dict = field(default_factory=lambda: {})

    def __iadd__(self, other):
        self.data.update(other.data)
        return self


class FilterCompositeVisitor(RootCompositeVisitor):

    def __init__(self, task_flow, prefix):
        super().__init__(task_flow, TuningVisitor, TunedParams, prefix=prefix)

    def load_tuned(self, tuned_params):
        tasks = self.task_flow.get_all_children()
        for path, task in tasks.items():
            task.get_decoder().load_tuned(tuned_params[path])
