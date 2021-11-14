from typing import List, Tuple

from dnn_cool.evaluation import EvaluationVisitor
from dnn_cool.external.autograd import IAutoGrad
from dnn_cool.filter import FilterVisitor
from dnn_cool.metrics import TorchMetric
from dnn_cool.tasks.base import IMinimal


class TaskForDevelopment(IMinimal):

    def __init__(self, task,
                 labels,
                 criterion,
                 per_sample_criterion,
                 available_func,
                 metrics: List[Tuple[str, TorchMetric]],
                 autograd: IAutoGrad,
                 precondition_func=None):
        self.name = task.get_name()
        self.labels = labels
        self.criterion = criterion
        self.per_sample_criterion = per_sample_criterion
        self.available_func = available_func
        self.metrics = metrics if metrics is not None else []
        self.task = task
        self.autograd = autograd
        self.precondition_func = precondition_func

    def get_name(self) -> str:
        return self.name

    def get_filter(self) -> FilterVisitor:
        return FilterVisitor(self, prefix='', autograd=self.autograd)

    def get_evaluator(self) -> EvaluationVisitor:
        return EvaluationVisitor(self, prefix='', autograd=self.autograd)

    def get_available_func(self):
        return self.available_func

    def get_criterion(self, prefix='', ctx=None):
        return self.criterion

    def get_per_sample_criterion(self, prefix='', ctx=None):
        return self.per_sample_criterion

    def get_labels(self):
        return self.labels

    def get_metrics(self):
        for i in range(len(self.metrics)):
            metric_name, metric = self.metrics[i]
            metric.bind_to_task(self.get_minimal())
        return self.metrics

    def get_minimal(self):
        return self.task
