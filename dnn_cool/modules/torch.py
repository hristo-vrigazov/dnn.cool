import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn

from dnn_cool.dsl import ICondition, IFlowTaskResult
from dnn_cool.modules.base import _copy_to_self, FeaturesDict
from dnn_cool.utils.base import dict_get_along_keys
from dnn_cool.utils.torch import to_broadcastable_shape


class SigmoidAndMSELoss(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        activated_output = self.sigmoid(output)
        return self.mse(activated_output, target)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ModuleDecorator(nn.Module):

    def __init__(self, task, prefix):
        super().__init__()
        self.prefix = prefix
        self.task_name = task.get_name()
        self.module = task.torch()
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()
        self.dropout_mc = task.get_dropout_mc()

    def forward(self, *args, **kwargs):
        key = self.prefix + self.task_name
        logits = self.module(*args, **kwargs)
        condition = LabelsAvailableCondition(key)
        if self.training:
            return LeafModuleOutput(path=key, logits=logits, precondition=condition,
                                    activated=None, decoded=None, dropout_samples=None)
        activated_logits = self.activation(logits) if self.activation is not None else logits
        decoded_logits = self.decoder(activated_logits) if self.decoder is not None else activated_logits
        condition = LabelsAvailableCondition(key)
        dropout_samples = None if self.dropout_mc is None else self.dropout_mc.create_samples(self.module,
                                                                                              self.activation,
                                                                                              *args, **kwargs)
        return LeafModuleOutput(path=key, logits=logits, precondition=condition,
                                activated=activated_logits, decoded=decoded_logits, dropout_samples=dropout_samples)


class Condition(ICondition):

    def get_precondition(self):
        raise NotImplementedError()

    def to_mask(self, data, shape):
        raise NotImplementedError()

    def __invert__(self):
        return NegatedCondition(self)

    def __and__(self, other):
        return AndCondition(self, other)


@dataclass
class LabelsAvailableCondition(Condition):
    path: str

    def get_precondition(self):
        return LabelsAvailableCondition(self.path)

    def to_mask(self, data, shape):
        if '_availability' in data:
            return data['_availability'][self.path]
        t = data.get(self.path)
        if t is None:
            raise ValueError(f'Path "{self.path}" leads to a None object.\n'
                             f'If you are trying to get treelib explanations, your model must be in eval mode.')
        return torch.ones(*shape, dtype=bool, device=t.device)


@dataclass()
class NegatedCondition(Condition):
    precondition: Condition

    def get_precondition(self):
        return self.precondition.get_precondition()

    def to_mask(self, data, shape):
        mask = self.precondition.to_mask(data, shape)
        precondition = self.get_precondition().to_mask(data, shape)
        mask[~precondition] = False
        mask[precondition] = ~mask[precondition]
        return mask


@dataclass
class LeafCondition(Condition):
    path: str

    def get_precondition(self):
        return LabelsAvailableCondition(self.path)

    def to_mask(self, data, shape):
        return data[self.path].clone()


@dataclass()
class NestedCondition(Condition):
    path: str
    parent: Condition

    def get_precondition(self):
        return self.parent

    def to_mask(self, data, shape):
        mask = data[self.path].clone()
        precondition = self.get_precondition().to_mask(data, shape)
        mask[~precondition] = False
        return mask


@dataclass()
class AndCondition(Condition):
    condition_one: Condition
    condition_two: Condition

    def get_precondition(self):
        return self.condition_one.get_precondition() & self.condition_two.get_precondition()

    def to_mask(self, data, shape):
        mask_one = self.condition_one.to_mask(data, shape)
        mask_two = self.condition_two.to_mask(data, shape)
        mask_one, mask_two = to_broadcastable_shape(mask_one, mask_two)
        return mask_one & mask_two


class IModuleOutput(IFlowTaskResult, ABC):
    pass


@dataclass
class LeafModuleOutput(IModuleOutput):
    path: str
    logits: torch.Tensor
    precondition: Condition
    activated: Optional[torch.Tensor]
    decoded: Optional[torch.Tensor]
    dropout_samples: Optional[torch.Tensor]

    def add_to_composite(self, composite_module_output):
        composite_module_output.logits[self.path] = self.logits
        composite_module_output.activated[self.path] = self.activated
        composite_module_output.decoded[self.path] = self.decoded
        composite_module_output.preconditions[self.path] = self.precondition
        composite_module_output.dropout_samples[self.path] = self.dropout_samples

    def __or__(self, precondition: Condition):
        """
        Note: this has the meaning of a a precondition, not boolean or.
        :param precondition: precondition
        :return: self
        """
        self.precondition &= precondition
        return self


@dataclass
class CompositeModuleOutput(IModuleOutput):
    training: bool
    gt: Dict[str, torch.Tensor]
    prefix: str
    logits: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    activated: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    decoded: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    preconditions: Dict[str, Condition] = field(default_factory=lambda: {})
    # A number of predictions for the same tasks, with different parts dropped out
    dropout_samples: Dict[str, torch.Tensor] = field(default_factory=lambda: {})

    def add_to_composite(self, other):
        _copy_to_self(self.logits, other.logits)
        _copy_to_self(self.activated, other.activated)
        _copy_to_self(self.decoded, other.decoded)
        _copy_to_self(self.preconditions, other.preconditions)

    def __iadd__(self, other):
        other.add_to_composite(self)
        return self

    def __getattr__(self, item):
        full_path = self.prefix + item
        parent_precondition = self.preconditions.get(full_path)
        if parent_precondition is None:
            return LeafCondition(path=full_path)
        return NestedCondition(path=full_path, parent=parent_precondition)

    def __or__(self, precondition: Condition):
        """
        Note: this has the meaning of a a precondition, not boolean or.
        :param precondition: precondition tensor
        :return: self
        """
        for key, value in self.preconditions.items():
            current_precondition = self.preconditions.get(key)
            if current_precondition is None:
                self.preconditions[key] = precondition
            else:
                self.preconditions[key] &= precondition
        return self

    def reduce(self):
        if len(self.prefix) != 0:
            # if this is not root, do not reduce yet.
            return self
        inference_without_gt = self.gt is None
        preconditions_source = self.decoded if inference_without_gt else self.gt
        res = self.decoded if inference_without_gt else self.logits

        if inference_without_gt and not self.training:
            for key, value in self.preconditions.items():
                if value is not None:
                    default_shape = self.logits[key].shape[:-1]
                    self.preconditions[key] = value.to_mask(preconditions_source, default_shape)
            return self
        for key, value in self.preconditions.items():
            if value is not None:
                default_shape = self.logits[key].shape[:-1]
                res[f'precondition|{key}'] = value.to_mask(preconditions_source, default_shape)
        if not inference_without_gt:
            res['gt'] = self.gt
        return res

    def __getitem__(self, item):
        if isinstance(item, str):
            return LeafModuleOutput(
                path=item,
                logits=self.logits[item],
                precondition=self.preconditions[item],
                activated=self.activated[item],
                decoded=self.decoded[item],
                dropout_samples=self.dropout_samples[item]
            )
        return CompositeModuleOutput(
            training=self.training,
            gt=self.gt,
            prefix=self.prefix,
            logits=dict_get_along_keys(self.logits, item),
            activated=dict_get_along_keys(self.activated, item),
            decoded=dict_get_along_keys(self.decoded, item),
            preconditions=self.preconditions,
            dropout_samples=dict_get_along_keys(self.dropout_samples, item),
        )

    def __repr__(self):
        res = object.__repr__(self) + os.linesep
        fields = ['logits', 'activated', 'decoded']
        for f in fields:
            for k, v in getattr(self, f).items():
                res += f'{f} {k}: {v.shape}{os.linesep}'
        return res


class TaskFlowModule(nn.Module):

    def __init__(self, task_flow, prefix=''):
        super().__init__()
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We wi.ll then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()
        self.prefix = prefix

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = ModuleDecorator(task, prefix)
            else:
                instance = TaskFlowModule(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def forward(self, x):
        if isinstance(x, FeaturesDict):
            x = x.data

        out = CompositeModuleOutput(training=self.training, gt=x.get('gt'), prefix=self.prefix)
        composite_module_output = self.flow(self, FeaturesDict(x), out)
        outputs = composite_module_output.reduce()
        return outputs

    def load_tuned(self, tuned_params):
        decoders = self._get_all_decoders()
        for key, decoder in decoders.items():
            decoder.load_tuned(tuned_params[key])

    def _get_all_decoders(self):
        decoders = {}
        for task_name, task in self._task_flow.tasks.items():
            if task.has_children():
                attr: TaskFlowModule = getattr(self, task_name)
                decoders.update(attr._get_all_decoders())
            else:
                decoders[self.prefix + task_name] = task.get_decoder()
        return decoders
