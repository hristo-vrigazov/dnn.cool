import torch
from torch import nn

from dnn_cool.dsl import IFeaturesDict, IOut, ICondition, IFlowTaskResult
from dnn_cool.utils import any_value


class ReducedPerSample(nn.Module):

    def __init__(self, loss, reduction):
        super().__init__()
        self.loss = loss
        self.reduction = reduction

    def forward(self, *args, **kwargs):
        loss_results = self.loss(*args, **kwargs)
        n_dims = len(loss_results.shape)
        if n_dims > 1:
            dims_to_reduce = tuple(range(1, n_dims))
            return self.reduction(loss_results, dim=dims_to_reduce, keepdim=True)
        return loss_results


class LossFlowData(IFeaturesDict):

    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    # To be compatible with the pipeline
    def __getattr__(self, item):
        return self


class LossItems(IOut, IFlowTaskResult, ICondition):

    def __init__(self, loss_items):
        self.loss_items = loss_items

    def __iadd__(self, other):
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
    if len(args) == 2:
        return LossFlowData(*args)

    for arg in args:
        if isinstance(arg, LossFlowData):
            return arg

    for arg in kwargs.values():
        if isinstance(arg, LossFlowData):
            return arg


def squeeze_if_needed(tensor):
    if len(tensor.shape) > 2:
        raise ValueError(f'Trying to squeeze the second dimension out of a tensor with shape: {tensor.shape}')
    if len(tensor.shape) == 2:
        return tensor[:, 0]
    return tensor


class BaseMetricDecorator(nn.Module):

    def __init__(self, task_name, available_func, prefix, metric):
        super().__init__()
        self.task_name = task_name
        self.available = available_func
        self.prefix = prefix
        self.metric = metric

    def forward(self, *args, **kwargs):
        loss_flow_data = get_flow_data(*args, **kwargs)
        return self.compute_with_precondition(loss_flow_data, self.metric)

    def compute_with_precondition(self, loss_flow_data, metric):
        key = self.prefix + self.task_name
        outputs = loss_flow_data.outputs[key]
        precondition = loss_flow_data.outputs[f'precondition|{key}']
        targets = loss_flow_data.targets[key]
        loss_items = torch.zeros(1, dtype=outputs.dtype, device=outputs.device)
        if precondition.sum() == 0:
            return loss_items
        precondition = squeeze_if_needed(precondition)
        metric_res = metric(outputs[precondition], targets[precondition])
        return self.postprocess_results(loss_items, metric_res, precondition)

    def postprocess_results(self, loss_items, metric_res, precondition):
        return metric_res


class TaskLossDecorator(BaseMetricDecorator):

    def __init__(self, task_name, available_func, prefix, loss):
        super().__init__(task_name, available_func, prefix, loss)

    def postprocess_results(self, loss_items, metric_res, precondition):
        return LossItems(metric_res)


class TaskFlowLoss(nn.Module):

    def __init__(self, task_flow, prefix=''):
        super().__init__()
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = TaskLossDecorator(task.get_name(), task.get_available_func(), prefix, task.get_loss())
            else:
                instance = TaskFlowLoss(task, prefix=f'{prefix}{task.get_name()}.')

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
        loss_items = torch.zeros(1, dtype=value.dtype, device=value.device)
        flow_result = self.flow(self, LossFlowData(outputs, targets), LossItems(loss_items))

        if not is_root:
            return LossItems(flow_result.loss_items)

        return flow_result.loss_items

    def get_leaf_losses(self):
        all_losses = {}
        for key, task in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            if task.has_children():
                all_losses.update(child_loss.get_leaf_losses())
            else:
                path = child_loss.prefix + child_loss.task_name
                all_losses[path] = child_loss
        return all_losses

    def get_metrics(self):
        all_metrics = []
        for key, task in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            if task.has_children():
                all_metrics += child_loss.get_metrics()
            else:
                for metric_name, metric in task.get_metrics():
                    metric_decorator = BaseMetricDecorator(task.get_name(),
                                                           task.get_available_func(),
                                                           child_loss.prefix,
                                                           metric)
                    all_metrics.append((metric_name, metric_decorator))
        return all_metrics

    def catalyst_callbacks(self):
        from catalyst.core import MetricCallback
        callbacks = []
        for path, loss in self.get_leaf_losses().items():
            metric_decorator = BaseMetricDecorator(loss.task_name,
                                                   loss.available,
                                                   loss.prefix,
                                                   loss.metric)
            callbacks.append(MetricCallback(f'loss_{path}', metric_decorator))
        for metric_name, metric_decorator in self.get_metrics():
            full_name = f'{metric_name}_{metric_decorator.prefix}{metric_decorator.task_name}'
            callback = MetricCallback(full_name, metric_decorator)
            callbacks.append(callback)
        return callbacks


class TaskFlowLossPerSample(nn.Module):

    def __init__(self, task_flow, prefix=''):
        super().__init__()

        self._all_children = task_flow.get_all_children(prefix=prefix)
        self._all_losses = self._collect_leaf_losses_per_sample()

    def forward(self, outputs, targets):
        res = {}
        value = any_value(outputs)
        bs = len(value)
        overall_loss_items = torch.zeros(bs, device=value.device, dtype=value.dtype)
        for path, loss in self._all_losses.items():
            loss_items = loss(outputs, targets).loss_items
            res[path] = loss_items.squeeze(dim=-1)
            indices = torch.arange(bs, device=value.device)
            precondition = outputs[f'precondition|{path}']
            axes = tuple(range(1, len(precondition.shape)))
            if len(axes) > 0:
                precondition = precondition.sum(axis=axes) > 0
            res[f'indices|{path}'] = indices[precondition]
            overall_loss_items[precondition] += res[path]
        res['overall'] = overall_loss_items
        res['indices|overall'] = torch.arange(bs, device=value.device)
        return res

    def get_leaf_losses_per_sample(self):
        return self._all_losses

    def _collect_leaf_losses_per_sample(self):
        all_losses = {}
        for path, task in self._all_children.items():
            if '.' not in path:
                prefix = ''
            else:
                prefix = path.rsplit('.', 1)[0] + '.'
            all_losses[path] = TaskLossDecorator(task.get_name(),
                                                 task.get_available_func(),
                                                 prefix,
                                                 task.get_per_sample_loss())
        return all_losses
