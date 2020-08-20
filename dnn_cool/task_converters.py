from typing import Optional, Tuple, Callable

from torch import nn

from dnn_cool.decoders import Decoder
from dnn_cool.metrics import TorchMetric


class To:
    """Syntactic sugar for task converters. You can achieve the same through functools partial, but here you can
    give a supplier for some fields, so that the creation of the respective field is delayed."""

    def __init__(self,
                 task_cls,
                 module_supplier: Callable[[], nn.Module] = None,
                 loss_supplier: Callable[[], nn.Module] = None,
                 per_sample_loss_supplier: Callable[[], nn.Module] = None,
                 decoder_supplier: Callable[[], Decoder] = None,
                 available_func=None,
                 inputs=None,
                 activation: Optional[nn.Module] = None,
                 metrics: Tuple[str, TorchMetric] = None):
        self.task_cls = task_cls
        self.module_supplier = module_supplier
        self.loss_supplier = loss_supplier
        self.per_sample_loss_supplier = per_sample_loss_supplier
        self.decoder_supplier = decoder_supplier
        self.available_func = available_func
        self.inputs = inputs
        self.activation = activation
        self.metrics = metrics

    def __call__(self, name, labels):
        kwargs = {}
        if self.module_supplier is not None:
            kwargs['module'] = self.module_supplier()
        if self.loss_supplier is not None:
            kwargs['loss'] = self.loss_supplier()
        if self.per_sample_loss_supplier is not None:
            kwargs['per_sample_loss'] = self.per_sample_loss_supplier()
        if self.decoder_supplier is not None:
            kwargs['decoder'] = self.decoder_supplier()
        if self.available_func is not None:
            kwargs['available_func'] = self.available_func
        if self.inputs is not None:
            kwargs['inputs'] = self.inputs
        if self.activation is not None:
            kwargs['activation'] = self.activation

        return self.task_cls(name=name, labels=labels, **kwargs)
