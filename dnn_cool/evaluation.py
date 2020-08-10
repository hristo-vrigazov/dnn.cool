from dataclasses import dataclass, field
from typing import Dict

from dnn_cool.visitors import LeafVisitor, VisitorOut, RootCompositeVisitor


class EvaluationVisitor(LeafVisitor):

    def __init__(self, task, prefix):
        super().__init__(task, prefix)

    def full_result(self, preds, targets):
        return EvaluationResults({})

    def empty_result(self):
        return EvaluationResults({})

    def preconditioned_result(self, preds, targets):
        return EvaluationResults({})


@dataclass
class EvaluationResults(VisitorOut):
    data: Dict = field(default_factory=lambda: {})

    def __iadd__(self, other):
        self.data.update(other.data)
        return self

    def reduce(self):
        return self.data


class EvaluationCompositeVisitor(RootCompositeVisitor):

    def __init__(self, task_flow, prefix):
        super().__init__(task_flow, EvaluationVisitor, EvaluationResults, prefix=prefix)

    def load_tuned(self, tuned_params):
        tasks = self.task_flow.get_all_children()
        for path, task in tasks.items():
            task.get_decoder().load_tuned(tuned_params[path])
