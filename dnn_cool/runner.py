from collections import OrderedDict
from pathlib import Path

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, InferCallback, State
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters
from dnn_cool.task_flow import TaskFlow
from torch import optim

from functools import partial
from time import time

from dnn_cool.utils import TransformedSubset


class InferDictCallback(InferCallback):

    def __init__(self, out_key='logits', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_key = out_key

    def on_batch_end(self, state: State):
        dct = state.batch_out[self.out_key]
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            self.predictions[key].append(value)


class DnnCoolSupervisedRunner(SupervisedRunner):

    def __init__(self, project, early_stop: bool = True):
        super().__init__()
        self.task_flow = project.get_full_flow()

        self.default_criterion = self.task_flow.get_loss()
        self.default_callbacks = self.default_criterion.catalyst_callbacks()
        self.default_optimizer = partial(optim.AdamW, lr=1e-4)
        self.default_scheduler = ReduceLROnPlateau
        self.project_dir: Path = project.project_dir
        self.project_dir.mkdir(exist_ok=True)
        self.default_logdir = f'./logdir_{self.task_flow.get_name()}_{time()}'

        if early_stop:
            self.default_callbacks.append(EarlyStoppingCallback(patience=5))

        self.train_test_val_indices = project.train_test_val_indices
        self.tensor_loggers = project.converters.tensorboard_converters

    def train(self, *args, **kwargs):
        kwargs['criterion'] = kwargs.get('criterion', self.default_criterion)
        kwargs['callbacks'] = kwargs.get('callbacks', self.default_callbacks)

        if not 'optimizer' in kwargs:
            model = kwargs['model']
            optimizable_params = filter(lambda p: p.requires_grad, model.parameters())
            kwargs['optimizer'] = self.default_optimizer(params=optimizable_params)

        if not 'scheduler' in kwargs:
            kwargs['scheduler'] = self.default_scheduler(kwargs['optimizer'])

        if not 'logdir' in kwargs:
            kwargs['logdir'] = self.default_logdir
        kwargs['logdir'] = self.project_dir / kwargs['logdir']
        kwargs['num_epochs'] = kwargs.get('num_epochs', 50)

        if not 'loaders' in kwargs:
            kwargs['loaders'] = self.get_default_loaders()

        super().train(*args, **kwargs)

    def infer(self, *args, **kwargs):
        default_loaders = OrderedDict({'infer': self.get_default_loaders()['valid']})
        kwargs['loaders'] = kwargs.get('loaders', default_loaders)

        logdir = self.project_dir / Path(kwargs.get('logdir', self.default_logdir))
        tensorboard_converters = TensorboardConverters(
            logdir=logdir,
            tensorboard_loggers=self.tensor_loggers,
            datasets=kwargs.get('datasets', self.get_default_datasets())
        )

        interpretation_callback = InterpretationCallback(self.task_flow, tensorboard_converters)
        default_callbacks = OrderedDict([("interpretation", interpretation_callback),
                                         ("inference", InferDictCallback())])
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)
        kwargs.pop("logdir", None)
        super().infer(*args, **kwargs)
        results = kwargs['callbacks']['inference'].predictions
        interpretation = kwargs['callbacks']['interpretation'].interpretations
        return results, interpretation

    def get_default_loaders(self):
        datasets = self.get_default_datasets()
        train_dataset = datasets['train']
        val_dataset = datasets['valid']
        train_loader = DataLoader(train_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=False)
        loaders = OrderedDict({
            'train': train_loader,
            'valid': val_loader
        })
        return loaders

    def get_default_datasets(self):
        dataset = self.task_flow.get_dataset()
        if self.train_test_val_indices is None:
            raise ValueError(f'You must supply either a `loaders` parameter, or give `train_test_val_indices` via'
                             f'constructor.')
        train_indices, test_indices, val_indices = self.train_test_val_indices
        train_dataset = TransformedSubset(dataset, train_indices)
        val_dataset = TransformedSubset(dataset, val_indices)
        test_dataset = TransformedSubset(dataset, test_indices)
        return {
            'train': train_dataset,
            'valid': val_dataset,
            'test': test_dataset,
            'infer': val_dataset
        }

    def batch_to_device(self, batch, device):
        return super()._batch2device(batch, device)

    def batch_to_model_device(self, batch, model):
        return super()._batch2device(batch, next(model.parameters()).device)
