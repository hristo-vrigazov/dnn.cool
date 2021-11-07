from dataclasses import dataclass, field
from typing import List

import pandas as pd

from dnn_cool.visitors import LeafVisitor, VisitorOut, RootCompositeVisitor


class EvaluationVisitor(LeafVisitor):

    def __init__(self, task, prefix, autograd):
        super().__init__(task, prefix, autograd)
        self.metrics = task.get_metrics()

    def empty_result(self):
        return EvaluationResults([])

    def preconditioned_result(self, preds, targets):
        return self.compute_metrics(preds, targets)

    def compute_metrics(self, preds, targets):
        res = []
        for metric_name, metric in self.metrics:
            # No activation, since preds is already activated
            metric_res = metric(preds, targets, activate=False)
            if isinstance(metric_res, dict):
                for key, value in metric_res.items():
                    res.append(self.create_evaluation_record(f'{metric_name}_{key}', value, targets))
            else:
                res.append(self.create_evaluation_record(metric_name, metric_res, targets))
        return EvaluationResults(res)

    def create_evaluation_record(self, metric_name, metric_res, targets):
        metric_res = self.autograd.get_single_float(metric_res)
        return {
            'task_path': self.path,
            'metric_name': metric_name,
            'metric_res': metric_res,
            'num_samples': len(targets),
        }


@dataclass
class EvaluationResults(VisitorOut):
    data: List = field(default_factory=lambda: [])

    def __iadd__(self, other):
        self.data += other.data
        return self

    def reduce(self):
        return pd.DataFrame(self.data)


class EvaluationCompositeVisitor(RootCompositeVisitor):

    def __init__(self, task_flow, prefix, autograd):
        super().__init__(task_flow, EvaluationVisitor, EvaluationResults, prefix=prefix, autograd=autograd)

    def load_tuned(self, tuned_params):
        tasks = self.task_flow.get_all_children()
        for path, task in tasks.items():
            task.get_decoder().load_tuned(tuned_params[path])
