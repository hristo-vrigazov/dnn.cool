import torch
from torch import nn

from dnn_cool.utils import any_value


class LossFlowData:

    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    # To be compatible with the pipeline
    def __getattr__(self, item):
        return self


class LossItems:

    def __init__(self, loss_items):
        self.loss_items = loss_items

    def __add__(self, other):
        return LossItems(self.loss_items + other.loss_items)

    # None of the methods below modify the state. They are here
    # to be compatible with the pipeline
    def __getattr__(self, item):
        return self

    def __or__(self, result):
        return self

    def __invert__(self):
        return self

    def __and__(self, other):
        return self


def get_flow_data(*args, **kwargs):
    is_callback_invoked = len(args) == 2
    if is_callback_invoked:
        return is_callback_invoked, LossFlowData(*args)

    for arg in args:
        if isinstance(arg, LossFlowData):
            return is_callback_invoked, arg

    for arg in kwargs.values():
        if isinstance(arg, LossFlowData):
            return is_callback_invoked, arg


def squeeze_if_needed(tensor):
    if len(tensor.shape) > 2:
        raise ValueError(f'Trying to squeeze the second dimension out of a tensor with shape: {tensor.shape}')
    if len(tensor.shape) == 2:
        return tensor.squeeze(dim=1)
    return tensor


class BaseMetricDecorator(nn.Module):

    def __init__(self, task, prefix, metric):
        super().__init__()
        self.task_name = task.get_name()
        self.available = task.get_available_func()
        self.prefix = prefix
        self.metric = metric

    def forward(self, *args, **kwargs):
        is_callback_invoked, loss_flow_data = get_flow_data(*args, **kwargs)
        loss_items = self.read_vectors(loss_flow_data, self.metric)

        if is_callback_invoked:
            return loss_items.mean()

        return LossItems(loss_items)

    def read_vectors(self, loss_flow_data, metric):
        key = self.prefix + self.task_name
        outputs = loss_flow_data.outputs[key]
        precondition = loss_flow_data.outputs.get(f'precondition|{key}', None)
        targets = loss_flow_data.targets[key]
        if self.available is not None:
            available = self.available(targets)
            if precondition is None:
                precondition = available
            else:
                precondition &= available
        n = len(outputs)
        loss_items = torch.zeros(n, 1, dtype=outputs.dtype, device=outputs.device)
        if precondition is None:
            return metric(outputs, targets)
        if precondition.sum() == 0:
            return loss_items
        precondition = squeeze_if_needed(precondition)
        metric_res = metric(outputs[precondition], targets[precondition])
        return self.postprocess_results(loss_items, metric_res, precondition)

    def postprocess_results(self, loss_items, metric_res, precondition):
        raise NotImplementedError()


class TaskLossDecorator(BaseMetricDecorator):

    def __init__(self, task, child_reduction, prefix):
        loss = task.get_loss() if child_reduction == 'mean' else task.get_per_sample_loss()
        super().__init__(task, prefix, loss)

    def postprocess_results(self, loss_items, metric_res, precondition):
        if len(metric_res.shape) == 1:
            metric_res = metric_res.unsqueeze(dim=1)
        loss_items[precondition] = metric_res
        return loss_items


class MetricLossDecorator(BaseMetricDecorator):

    def __init__(self, task, prefix, metric):
        super().__init__(task, prefix, metric)

    def postprocess_results(self, loss_items, metric_res, precondition):
        return metric_res


class TaskFlowLoss(nn.Module):

    def __init__(self, task_flow, parent_reduction, child_reduction, prefix=''):
        super().__init__()
        self.parent_reduction = parent_reduction
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = TaskLossDecorator(task, child_reduction, prefix)
            else:
                instance = TaskFlowLoss(task, child_reduction, child_reduction, prefix=f'{prefix}{task.get_name()}.')

            setattr(self, key, instance)

    def forward(self, *args):
        """
        TaskFlowLoss can be invoked either by giving two arguments: (outputs, targets), or bby giving a single
        LossFlowData argument, which holds the outputs and the targets.
        :param args:
        :return:
        """
        is_root = len(args) == 2
        if is_root:
            outputs, targets = args
        else:
            loss_flow_data = args[0]
            outputs = loss_flow_data.outputs
            targets = loss_flow_data.targets

        value = any_value(outputs)
        n = len(value)
        loss_items = torch.zeros(n, 1, dtype=value.dtype, device=value.device)
        flow_result = self.flow(self, LossFlowData(outputs, targets), LossItems(loss_items))

        if not is_root:
            return LossItems(flow_result.loss_items)

        if self.parent_reduction == 'mean':
            return flow_result.loss_items.mean()

        return flow_result.loss_items

    def get_leaf_losses(self):
        all_losses = []
        for key, task in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            if task.has_children():
                all_losses += child_loss.get_leaf_losses()
            else:
                all_losses.append(child_loss)
        return all_losses

    def get_metrics(self):
        all_metrics = []
        for key, task in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            for metric_name, metric in task.get_metrics():
                if task.has_children():
                    all_metrics += child_loss.get_metrics()
                else:
                    metric_decorator = MetricLossDecorator(task, child_loss.prefix, metric)
                    all_metrics.append((metric_name, metric_decorator))
        return all_metrics

    def catalyst_callbacks(self):
        from catalyst.core import MetricCallback
        callbacks = []
        for leaf_loss in self.get_leaf_losses():
            callbacks.append(MetricCallback(f'loss_{leaf_loss.prefix}{leaf_loss.task_name}', leaf_loss))
        for metric_name, metric in self.get_metrics():
            callbacks.append(MetricCallback(f'{metric_name}_{metric.prefix}{metric.task_name}', metric))
        return callbacks



