from torch.utils.data.dataset import Dataset


def discover_index_holder(*args, **kwargs):
    all_args = [*args, *kwargs.values()]

    for arg in all_args:
        if isinstance(arg, IndexHolder):
            return arg


class FlowDatasetDecorator:

    def __init__(self, task, prefix):
        self.task_name = task.get_name()
        self.prefix = prefix
        self.dataset = task.datasets()

    def __call__(self, *args, **kwargs):
        index_holder = discover_index_holder(*args, **kwargs)
        key = self.prefix + self.task_name
        return FlowDatasetDict(self.prefix, {
            key: self.dataset[index_holder.item]
        })


class IndexHolder:

    def __init__(self, item):
        self.item = item

    def __getattr__(self, item):
        return self


class FlowDatasetDict:

    def __init__(self, prefix, data):
        self.prefix = prefix
        self.data = data

    def __add__(self, other):
        for key, value in other.data.items():
            if key == 'gt':
                if not ('gt' in self.data):
                    self.data['gt'] = {}
                self.data['gt'].update(other.data['gt'])
            else:
                self.data[key] = value
        return self

    def __getattr__(self, item):
        return FlowDatasetDict(self.prefix, {
            'key': self.prefix + item,
            'precondition': self.data[self.prefix + item]
        })

    def __or__(self, other):
        gt_dict = {}
        X, y = other.data['precondition']
        gt_dict[other.data['key']] = y.bool()
        if not ('gt' in self.data):
            self.data['gt'] = {}
        self.data['gt'].update(gt_dict)
        return self

    def __invert__(self):
        X, y = self.data['precondition']
        new_data = {
            'key': self.data['key'],
            'precondition': (X, y.bool())
        }
        return FlowDatasetDict(self.prefix, new_data)

    def to_input_target(self):
        X = {}
        y = {}
        for key, value in self.data.items():
            if key == 'gt':
                X[key] = value
                continue
            inputs, targets = value
            # TODO: multi input processing
            X['features'] = inputs
            y[key] = targets
        return X, y


class FlowDataset(Dataset):

    def __init__(self, task_flow, prefix=''):
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.__class__.flow

        self.n = None
        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = FlowDatasetDecorator(task, prefix)
                self.n = len(instance.dataset)
            else:
                instance = FlowDataset(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)
        self.prefix = prefix

    def __getitem__(self, item):
        flow_dataset_dict = self.flow(self, IndexHolder(item), FlowDatasetDict(self.prefix, {}))
        return flow_dataset_dict.to_input_target()

    def __len__(self):
        return self.n

    def __call__(self, *args, **kwargs):
        index_holder = discover_index_holder(*args, **kwargs)
        return self.flow(self, index_holder, FlowDatasetDict(self.prefix, {}))
