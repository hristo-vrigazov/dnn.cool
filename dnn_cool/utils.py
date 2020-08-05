import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def any_value(outputs):
    for key, value in outputs.items():
        if not key.startswith('precondition'):
            return value


def torch_split_dataset(dataset, test_size=0.2, random_state=None):
    train_indices, test_indices = split_dataset(dataset, test_size, random_state=random_state)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def split_dataset(dataset, test_size=0.2, random_state=None):
    X, y = np.arange(len(dataset)), np.arange(len(dataset))
    train_indices, test_indices, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_indices, test_indices


def train_test_val_split(df, random_state=None):
    train_indices, val_indices = split_dataset(df, test_size=0.2, random_state=random_state)
    test_indices, val_indices = split_dataset(val_indices, test_size=0.5, random_state=random_state)
    return train_indices, test_indices, val_indices


class TransformedSubset(Dataset):
    r"""
    Subset of a dataset at specified indices, with additionally applied transforms.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transforms=None):
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms

    def __getitem__(self, idx):
        r = self.dataset[self.indices[idx]]
        if self.transforms is None:
            return r
        return self.transforms(r)

    def __len__(self):
        return len(self.indices)
