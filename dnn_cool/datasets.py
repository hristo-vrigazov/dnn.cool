from torch.utils.data.dataset import Dataset


class FlowDatasetDecorator:

    def __init__(self, task, prefix):
        pass


class IndexHolder:

    def __init__(self, item):
        self.item = item


class FlowDatasetDict:

    def __init__(self, item):
        pass


class FlowDataset(Dataset):

    def __init__(self, task_flow, prefix=''):
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.__class__.flow

        for key, task in task_flow.tasks.items():
            if not task.has_children():
                instance = FlowDatasetDecorator(task, prefix)
            else:
                instance = FlowDataset(task, prefix=f'{prefix}{task.get_name()}.')
            setattr(self, key, instance)

    def __getitem__(self, item):
        flow_dataset_dict = self.flow(self, IndexHolder(item), FlowDatasetDict({}))

    def __len__(self):
        pass