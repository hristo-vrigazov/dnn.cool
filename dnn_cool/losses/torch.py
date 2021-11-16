import torch
from catalyst.callbacks import BatchMetricCallback
from torch import nn

from dnn_cool.external.autograd import squeeze_last_dim_if_needed
from dnn_cool.losses.base import DeviceReducingCache, CriterionFlowData, LossItems, args_are_dicts
from dnn_cool.utils.base import any_value


class ReducedPerSample(nn.Module):

    def __init__(self, loss, from_dim=1):
        super().__init__()
        self.loss = loss
        self.from_dim = from_dim

    def forward(self, outputs, targets, precondition):
        loss_results = torch.zeros_like(targets, dtype=outputs.dtype)
        loss_results[precondition] = self.loss(outputs[precondition], targets[precondition])
        n_dims = len(loss_results.shape)
        if not (n_dims > 1):
            return loss_results[precondition]
        dims = tuple(range(self.from_dim, n_dims))
        total = loss_results.sum(dim=dims, keepdim=False)
        dims = tuple(range(self.from_dim,  len(precondition.shape)))
        if len(precondition.shape) > 1:
            precondition_sum = precondition.sum(dim=dims, keepdim=False)
            precondition = precondition_sum > 0
        else:
            precondition_sum = precondition.type(total.dtype)
        res = total / precondition_sum
        return res[precondition]


class BaseMetricDecorator(nn.Module):

    def __init__(self, task_name, prefix, metric, metric_name, ctx):
        super().__init__()
        self.task_name = task_name
        self.prefix = prefix
        self.metric = metric
        self.metric_name = metric_name
        self.ctx = ctx
        self._device_reducing_cache = DeviceReducingCache(task_name=task_name,
                                                          prefix=prefix,
                                                          metric=metric,
                                                          metric_name=metric_name,
                                                          ctx=ctx)

    def forward(self, *args, **kwargs):
        loss_flow_data = CriterionFlowData.from_args(*args, **kwargs)
        return self.postprocess_results(self.compute_with_precondition(loss_flow_data))

    def compute_with_precondition(self, loss_flow_data):
        key = self.prefix + self.task_name
        result_from_device_reducing = self._device_reducing_cache(key, loss_flow_data.outputs)
        if result_from_device_reducing is not None:
            return result_from_device_reducing
        outputs = loss_flow_data.outputs[key]
        precondition = loss_flow_data.outputs[f'precondition|{key}']
        targets = loss_flow_data.targets[key]
        if precondition.sum() == 0:
            return self.handle_empty_precondition(outputs)
        precondition = squeeze_last_dim_if_needed(precondition)
        if '_per_' in self.metric_name:
            metric_res = self.metric(outputs, targets, precondition)
        else:
            metric_res = self.metric(outputs[precondition], targets[precondition])
        return metric_res

    def handle_empty_precondition(self, outputs):
        if not hasattr(self.metric, 'empty_precondition_result'):
            return torch.tensor(0., dtype=outputs.dtype, device=outputs.device)
        return self.metric.empty_precondition_result()

    def postprocess_results(self, metric_res):
        return metric_res


class TaskCriterionDecorator(BaseMetricDecorator):

    def __init__(self, task_name, prefix, loss, metric_name, ctx):
        super().__init__(task_name, prefix, loss, metric_name, ctx)

    def postprocess_results(self, metric_res):
        return LossItems(metric_res)


class TaskFlowCriterion(nn.Module):

    def __init__(self, task_flow, prefix='', ctx=None):
        super().__init__()
        if ctx is None:
            ctx = {}
        self.ctx = ctx
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()

        for key, task_flow_for_development in task_flow.tasks.items():
            task_flow = task_flow_for_development.task
            if not task_flow.has_children():
                instance = TaskCriterionDecorator(task_flow_for_development.get_name(),
                                                  prefix,
                                                  task_flow_for_development.get_criterion(),
                                                  'loss',
                                                  self.ctx)
            else:
                instance = TaskFlowCriterion(task_flow_for_development,
                                             prefix=f'{prefix}{task_flow_for_development.get_name()}.',
                                             ctx=self.ctx)
            setattr(self, key, instance)

        self.prefix = prefix
        self.task_name = self._task_flow.get_name()
        self._device_reducing_cache = DeviceReducingCache(task_name=self._task_flow.get_name(),
                                                          prefix=prefix,
                                                          metric=self.flow,
                                                          metric_name='loss',
                                                          ctx=ctx)

    def forward(self, *args):
        """
        TaskFlowLoss can be invoked either by giving two arguments: (outputs, targets), or bby giving a single
        LossFlowData argument, which holds the outputs and the targets.
        :param args:
        :return:
        """
        if args_are_dicts(args):
            outputs, targets = args
        else:
            loss_flow_data = args[0]
            outputs = loss_flow_data.outputs
            targets = loss_flow_data.targets

        value = any_value(outputs)
        key = self.prefix + self.task_name
        result_from_device_reducing = self._device_reducing_cache(key, outputs)
        if result_from_device_reducing is not None:
            return result_from_device_reducing
        loss_items = torch.zeros(1, dtype=value.dtype, device=value.device)
        flow_result = self.flow(self, CriterionFlowData(outputs, targets), LossItems(loss_items))

        is_root = self.prefix == ''
        if not is_root:
            return LossItems(flow_result.loss_items)

        return flow_result.loss_items

    def get_leaf_losses(self):
        all_losses = {}
        for key, task_flow_for_development in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            if task_flow_for_development.task.has_children():
                all_losses.update(child_loss.get_leaf_losses())
            else:
                path = child_loss.prefix + child_loss.task_name
                all_losses[path] = child_loss
        return all_losses

    def get_metrics(self):
        all_metrics = []
        for key, task_for_development in self._task_flow.tasks.items():
            child_loss = getattr(self, key)
            if task_for_development.task.has_children():
                all_metrics += child_loss.get_metrics()
            else:
                for metric_name, metric in task_for_development.get_metrics():
                    metric_decorator = BaseMetricDecorator(task_for_development.get_name(),
                                                           child_loss.prefix,
                                                           metric,
                                                           metric_name,
                                                           self.ctx)
                    all_metrics.append((metric_name, metric_decorator))
        return all_metrics

    def catalyst_callbacks(self):
        callbacks = []
        for path, loss in self.get_leaf_losses().items():
            metric_decorator = BaseMetricDecorator(loss.task_name,
                                                   loss.prefix,
                                                   loss.metric,
                                                   'loss',
                                                   self.ctx)
            callbacks.append(BatchMetricCallback(f'loss_{path}', metric_decorator))
        for metric_name, metric_decorator in self.get_metrics():
            full_name = f'{metric_name}_{metric_decorator.prefix}{metric_decorator.task_name}'
            callback = BatchMetricCallback(full_name, metric_decorator)
            callbacks.append(callback)
        return callbacks

    def get_device_reduced_ctx(self):
        return self.ctx


class TaskFlowLossPerSample(nn.Module):

    def __init__(self, task_flow, prefix='', ctx=None):
        super().__init__()
        if ctx is None:
            ctx = {}
        self.ctx = ctx
        self._all_children = task_flow.get_all_children(prefix=prefix)
        self._all_losses = self._collect_leaf_losses_per_sample()

    def forward(self, outputs, targets):
        res = {}
        value = any_value(outputs)
        device_n_key = f'_device|overall|_n'
        if device_n_key in outputs:
            bs = int(outputs[device_n_key].sum().item())
        else:
            bs = len(value)
        overall_loss_items = torch.zeros(bs, dtype=value.dtype)
        for path, loss in self._all_losses.items():
            loss_items = loss(outputs, targets).loss_items
            res[path] = loss_items
            device_key = f'_device|indices|{path}|loss_per_sample'
            if device_key in self.ctx:
                indices = self.ctx[device_key].detach().cpu()
                valid_indices_mask = indices >= 0
                res[f'indices|{path}'] = indices[valid_indices_mask]
                res[path] = res[path][valid_indices_mask]
                overall_loss_items[indices[valid_indices_mask]] += res[path]
            else:
                indices = torch.arange(bs, device=value.device)
                precondition = outputs[f'precondition|{path}']
                axes = tuple(range(1, len(precondition.shape)))
                if len(axes) > 0:
                    precondition = precondition.sum(axis=axes) > 0
                if precondition.sum() == 0:
                    # Since the metric decorator will add a 0 term to the loss, we have
                    # to mark the indices as well. This is later checked in the if above.
                    res[f'indices|{path}'] = torch.ones(1, dtype=indices.dtype, device=indices.device) * -1
                else:
                    res[f'indices|{path}'] = indices[precondition]
                    overall_loss_items[precondition] += res[path].cpu()
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
            all_losses[path] = TaskCriterionDecorator(task.get_name(),
                                                      prefix,
                                                      task.get_per_sample_criterion(ctx=self.ctx),
                                                      'loss_per_sample',
                                                      self.ctx)
        return all_losses


class LanguageModelCrossEntropyLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs, targets):
        # outputs is of shape (N, W, V) where N is batch size, W is number of tokens, V is number of tokens in vocab
        return self.ce(outputs.permute(0, 2, 1), targets)