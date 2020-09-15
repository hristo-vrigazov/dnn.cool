import torch

from dataclasses import dataclass
from torch.utils.data.dataset import Dataset

from dnn_cool.dsl import IFeaturesDict, IOut, ICondition, IFlowTask


class LeafTaskDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels), f'The provided inputs and labels are of different length: ' \
                                                     f'{len(self.inputs)} and {len(self.labels)}'

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


def discover_index_holder(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, IndexHolder):
            return arg


class FlowDatasetDecorator:

    def __init__(self, task, prefix, labels):
        self.task_name = task.get_name()
        self.available = task.get_available_func()(labels)
        self.prefix = prefix
        self.arr = labels

    def __call__(self, *args, **kwargs):
        index_holder = discover_index_holder(*args, **kwargs)
        key = self.prefix + self.task_name
        data = {
            key: self.arr[index_holder.item]
        }
        available = {
            key: self.available[index_holder.item]
        }
        return FlowDatasetDict(self.prefix, data, available)


class IndexHolder(IFeaturesDict):

    def __init__(self, item):
        self.item = item

    def __getattr__(self, item):
        return self


@dataclass
class FlowDatasetPrecondition(ICondition):
    prefix: str
    path: str
    precondition: torch.Tensor

    def __invert__(self):
        return self

    def __and__(self, other):
        return self


class FlowDatasetDict(IOut):

    def __init__(self, prefix, data=None, available=None):
        self.prefix = prefix
        self.data = data if data is not None else {}
        self.available = available if available is not None else {}
        self.gt = {}

    def __iadd__(self, other):
        for key, value in other.data.items():
            self.data[key] = value
        self.gt.update(other.gt)
        self.available.update(other.available)
        return self

    def __getattr__(self, item):
        return FlowDatasetPrecondition(prefix=self.prefix,
                                       path=self.prefix + item,
                                       precondition=self.data[self.prefix + item])

    def __or__(self, other: FlowDatasetPrecondition):
        self.gt.update({other.path: other.precondition.bool()})
        return self

    def to_dict(self, X):
        y = {}
        for key, value in self.data.items():
            targets = value
            y[key] = targets
        X['gt'] = self.gt
        X['gt']['_availability'] = self.available
        return X, y


class FlowDataset(Dataset, IFlowTask):

    def __init__(self, task_flow, prefix=''):
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()

        self.n = None
        for key, task in task_flow.tasks.items():
            if not task.has_children():
                labels_instance = FlowDatasetDecorator(task, prefix, task.get_labels())
                self.n = len(labels_instance.arr)
                setattr(self, key, labels_instance)
            else:
                instance = FlowDataset(task, prefix=f'{prefix}{task.get_name()}.')
                setattr(self, key, instance)
        self.prefix = prefix

    def __getitem__(self, item):
        """
        This method is going to execute the flow given in the constructor, passing around the index variable `item` in
        the class `IndexHolder`
        :param item:
        :return:
        """
        flow_dataset_dict = self.flow(self, IndexHolder(item), FlowDatasetDict(self.prefix, {}))
        inputs = self._task_flow.get_inputs()
        if inputs is None:
            raise ValueError(f'Cannot build a dataset, since the inputs are not provided. You have to provide them'
                             f' in the constructor of the TaskFlow class.')
        X = inputs[item]
        # X has to be a dict, because we have to attach gt.
        if not isinstance(X, dict):
            X = {
                'inputs': X
            }
        return flow_dataset_dict.to_dict(X)

    def __len__(self):
        return self.n

    def __call__(self, *args, **kwargs):
        index_holder = discover_index_holder(*args, **kwargs)
        return self.flow(self, index_holder, FlowDatasetDict(self.prefix))
