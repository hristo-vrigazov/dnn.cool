from dnn_cool.dsl import IFeaturesDict, IOut, IFlowTaskResult, ICondition


class CriterionFlowData(IFeaturesDict):

    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    # To be compatible with the pipeline
    def __getattr__(self, item):
        return self

    @classmethod
    def from_args(cls, *args, **kwargs):
        for arg in args:
            if isinstance(arg, CriterionFlowData):
                return arg

        for arg in kwargs.values():
            if isinstance(arg, CriterionFlowData):
                return arg

        if args_are_dicts(args):
            return cls(*args)


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


def args_are_dicts(args):
    return len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict)


def _already_reduced_per_device(outputs):
    keys = list(outputs.keys())
    if len(keys) == 0:
        return False
    return '_device' in keys[0]


class DeviceReducingCache:

    def __init__(self, task_name, prefix, metric, metric_name, ctx):
        self.metric = metric
        self.metric_name = metric_name
        self.prefix = prefix
        self.task_name = task_name
        self.ctx = ctx

    def __call__(self, key, outputs):
        result_from_device_reducing = self._compute_device_reduced(key, outputs)
        if result_from_device_reducing is not None:
            return result_from_device_reducing
        result_from_device_reducing = self._compute_device_reduced(key, self.ctx)
        if result_from_device_reducing is not None:
            return result_from_device_reducing

    def _compute_device_reduced(self, key, outputs):
        already_reduced_per_device = _already_reduced_per_device(outputs)
        if already_reduced_per_device:
            device_metric_key = f'_device|{key}|{self.metric_name}'
            if device_metric_key in outputs:
                # This means that the loss function has already been computed inside the nn.DataParallel model.
                return self.aggregate_device_result(outputs, device_metric_key)
            # Checks the same for multi-metrics
            device_metric_keys = self.discover_metric_keys(device_metric_key)
            if len(device_metric_keys) > 0:
                return self.aggregate_device_results(outputs, device_metric_keys)
        return None

    def discover_metric_keys(self, device_metric_key):
        if not hasattr(self.metric, 'empty_precondition_result'):
            return []
        metric_args = self.metric.empty_precondition_result().keys()
        return [f'{device_metric_key}_{metric_arg}' for metric_arg in metric_args]

    def aggregate_device_result(self, loss_flow_data_outputs, out_key):
        if self.metric_name == 'loss_per_sample':
            metric_per_gpu_results = loss_flow_data_outputs[out_key]
            return metric_per_gpu_results
        if out_key not in loss_flow_data_outputs:
            return None
        metric_per_gpu_results = loss_flow_data_outputs[out_key]
        metric_per_gpu_counts = loss_flow_data_outputs[f'_device|{self.prefix}{self.task_name}|_n']
        return (metric_per_gpu_results * metric_per_gpu_counts).sum() / metric_per_gpu_counts.sum()

    def aggregate_device_results(self, loss_flow_data_outputs, out_keys):
        res = {}
        for out_key in out_keys:
            metric_name, metric_arg = out_key.split('|')[-1].split('_')
            res[metric_arg] = self.aggregate_device_result(loss_flow_data_outputs, out_key)
            if res[metric_arg] is None:
                return None
        return res
