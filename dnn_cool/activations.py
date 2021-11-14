from dataclasses import dataclass, field
from typing import Dict

from dnn_cool.visitors import VisitorOut, LeafVisitor, RootCompositeVisitor


class ActivationVisitor(LeafVisitor):

    def __init__(self, task, prefix, autograd):
        super().__init__(task, prefix, autograd)

    def empty_result(self):
        return ActivationsFiltered({self.path: {}})

    def preconditioned_result(self, preds, targets):
        return ActivationsFiltered({self.path: preds})


@dataclass
class ActivationsFiltered(VisitorOut):
    data: Dict = field(default_factory=lambda: {})

    def __iadd__(self, other):
        self.data.update(other.data)
        return self

    def reduce(self):
        return self.data


class CompositeActivation(RootCompositeVisitor):

    def __init__(self, task_flow, prefix, autograd):
        super().__init__(task_flow, ActivationVisitor, ActivationsFiltered, prefix, autograd)

