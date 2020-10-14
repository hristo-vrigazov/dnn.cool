from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn

from dnn_cool.dsl import IFeaturesDict, IFlowTaskResult, ICondition
from dnn_cool.utils import to_broadcastable_shape


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


# noinspection PyBroadException
def find_arg_with_gt(args, is_kwargs):
    new_args = {} if is_kwargs else []
    gt = None
    for arg in args:
        if is_kwargs:
            arg = args[arg]
        try:
            gt = arg['gt']
            new_args.append(arg['value'])
        except:
            new_args.append(arg)
    return gt, new_args


def select_gt(a_gt, k_gt):
    if a_gt is None and k_gt is None:
        return {}

    if a_gt is None:
        return k_gt

    return a_gt


def find_gt_and_process_args_when_training(*args, **kwargs):
    a_gt, args = find_arg_with_gt(args, is_kwargs=False)
    k_gt, kwargs = find_arg_with_gt(kwargs, is_kwargs=True)
    return args, kwargs, select_gt(a_gt, k_gt)


class ModuleDecorator(nn.Module):

    def __init__(self, task, prefix):
        super().__init__()
        self.prefix = prefix
        self.task_name = task.get_name()
        self.module = task.torch()
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()

    def forward(self, *args, **kwargs):
        if self.training:
            args, kwargs, gt = find_gt_and_process_args_when_training(*args, **kwargs)
            decoded_logits = gt.get(self.task_name, None)
        else:
            decoded_logits = None

        logits = self.module(*args, **kwargs)
        activated_logits = self.activation(logits) if self.activation is not None else logits

        if decoded_logits is None:
            decoded_logits = self.decoder(activated_logits) if self.decoder is not None else activated_logits
        key = self.prefix + self.task_name
        condition = OnesCondition(key)
        return LeafModuleOutput(key, logits, activated_logits, decoded_logits, condition)


class Condition(ICondition):

    def get_precondition(self, data):
        raise NotImplementedError()

    def to_mask(self, data):
        raise NotImplementedError()

    def __invert__(self):
        return NegatedCondition(self)

    def __and__(self, other):
        return AndCondition(self, other)


@dataclass
class OnesCondition(Condition):
    path: str

    def get_precondition(self, data):
        return OnesCondition(self.path)

    def to_mask(self, data):
        if '_availability' in data:
            return data['_availability'][self.path]
        return torch.ones_like(data[self.path]).bool()


@dataclass()
class NegatedCondition(Condition):
    precondition: Condition

    def get_precondition(self, data):
        return self.precondition.get_precondition(data)

    def to_mask(self, data):
        mask = self.precondition.to_mask(data)
        precondition = self.get_precondition(data).to_mask(data)
        mask[~precondition] = False
        mask[precondition] = ~mask[precondition]
        return mask


@dataclass
class LeafCondition(Condition):
    path: str

    def get_precondition(self, data):
        return OnesCondition(self.path)

    def to_mask(self, data):
        return data[self.path].clone()


@dataclass()
class NestedCondition(Condition):
    path: str
    parent: Condition

    def get_precondition(self, data):
        return self.parent

    def to_mask(self, data):
        mask = data[self.path].clone()
        precondition = self.get_precondition(data).to_mask(data)
        mask[~precondition] = False
        return mask


@dataclass()
class AndCondition(Condition):
    condition_one: Condition
    condition_two: Condition

    def get_precondition(self, data):
        return self.condition_one.get_precondition(data) & self.condition_two.get_precondition(data)

    def to_mask(self, data):
        mask_one = self.condition_one.to_mask(data)
        mask_two = self.condition_two.to_mask(data)
        mask_one, mask_two = to_broadcastable_shape(mask_one, mask_two)
        return mask_one & mask_two


@dataclass
class LeafModuleOutput(IFlowTaskResult):
    path: str
    logits: torch.Tensor
    activated: torch.Tensor
    decoded: torch.Tensor
    precondition: Condition

    def add_to_composite(self, composite_module_output):
        composite_module_output.logits[self.path] = self.logits
        composite_module_output.activated[self.path] = self.activated
        composite_module_output.decoded[self.path] = self.decoded
        composite_module_output.preconditions[self.path] = self.precondition

    def __or__(self, precondition: Condition):
        """
        Note: this has the meaning of a a precondition, not boolean or.
        :param precondition: precondition
        :return: self
        """
        self.precondition = precondition
        return self


def _copy_to_self(self_arr, other_arr):
    for key, value in self_arr.items():
        assert key not in other_arr, f'The key {key} has been added twice in the same workflow!.'
        other_arr[key] = value


@dataclass
class CompositeModuleOutput(IFlowTaskResult):
    training: bool
    gt: Dict[str, torch.Tensor]
    prefix: str
    logits: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    activated: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    decoded: Dict[str, torch.Tensor] = field(default_factory=lambda: {})
    preconditions: Dict[str, Condition] = field(default_factory=lambda: {})

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
        if len(self.prefix) == 0:
            inference_without_gt = self.gt is None
            preconditions_source = self.decoded if inference_without_gt else self.gt
            res = self.decoded if inference_without_gt else self.logits

            if inference_without_gt and not self.training:
                for key, value in self.preconditions.items():
                    if value is not None:
                        self.preconditions[key] = value.to_mask(preconditions_source)
                return self
            for key, value in self.preconditions.items():
                if value is not None:
                    res[f'precondition|{key}'] = value.to_mask(preconditions_source)
            if not inference_without_gt:
                res['gt'] = self.gt
            return res
        return self


@dataclass
class FeaturesDict(IFeaturesDict):
    data: Dict

    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):
        return self.data[item]


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
