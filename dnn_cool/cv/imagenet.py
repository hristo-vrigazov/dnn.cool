import numpy as np
import torch


def check_if_should_divide(sample):
    if sample.dtype == np.uint8:
        return True
    if sample.dtype == torch.uint8:
        return True
    if sample.max() > 1.:
        return True
    return False


def check_if_should_transpose(sample):
    return sample.shape[-1] == 3


class ImageNetNormalizer:

    def __init__(self, should_divide_by_255=None, should_transpose=None):
        from torchvision import transforms
        self.should_divide_by_255 = should_divide_by_255
        self.should_transpose = should_transpose
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        if self.should_divide_by_255 is None:
            self.should_divide_by_255 = check_if_should_divide(sample)
        if self.should_transpose is None:
            self.should_transpose = check_if_should_transpose(sample)
        if self.should_divide_by_255:
            sample = sample / 255.
        X = torch.tensor(sample).float()
        if self.should_transpose:
            X = X.permute(2, 0, 1)
        X = self.normalize(X)
        return X