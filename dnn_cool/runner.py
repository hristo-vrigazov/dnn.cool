from collections import OrderedDict
from functools import partial
from pathlib import Path
from time import time

import numpy as np
import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, InferCallback, State
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters
from dnn_cool.utils import TransformedSubset


class InferDictCallback(InferCallback):

    def __init__(self, out_key='logits', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_key = out_key
        self.predictions = {}
        self.targets = {}

    def on_loader_start(self, state: State):
        self.predictions[state.loader_name] = {}
        self.targets[state.loader_name] = {}

    def on_batch_end(self, state: State):
        dct = state.batch_out[self.out_key]
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            if key not in self.predictions[state.loader_name]:
                self.predictions[state.loader_name][key] = []
            self.predictions[state.loader_name][key].append(value)

        targets = state.batch_in['targets']
        for key, value in targets.items():
            if key not in self.targets[state.loader_name]:
                self.targets[state.loader_name][key] = []
            self.targets[state.loader_name][key].append(value.detach().cpu().numpy())

    def on_loader_end(self, state: State):
        self.predictions[state.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions[state.loader_name].items()
        }

        self.targets[state.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.targets[state.loader_name].items()
        }


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
            datasets, kwargs['loaders'] = self.get_default_loaders()

        default_callbacks = [self.get_interpretation_callback(**kwargs)] + self.default_callbacks
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)

        super().train(*args, **kwargs)

    def infer(self, *args, **kwargs):
        default_datasets, default_loaders = self.get_default_loaders(shuffle_train=False)
        kwargs['loaders'] = kwargs.get('loaders', default_loaders)
        kwargs['datasets'] = kwargs.get('datasets', default_datasets)

        logdir = self.project_dir / Path(kwargs.get('logdir', self.default_logdir))
        kwargs['logdir'] = logdir
        interpretation_callback = self.get_interpretation_callback(**kwargs)
        default_callbacks = OrderedDict([("interpretation", interpretation_callback),
                                         ("inference", InferDictCallback())])
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)
        kwargs.pop("logdir", None)
        del kwargs['datasets']
        super().infer(*args, **kwargs)
        results = kwargs['callbacks']['inference'].predictions
        targets = kwargs['callbacks']['inference'].targets
        interpretation = kwargs['callbacks']['interpretation'].interpretations

        out_dir = self.project_dir / 'infer'
        out_dir.mkdir(exist_ok=True)
        torch.save(results, out_dir / 'logits.pkl')
        torch.save(targets, out_dir / 'targets.pkl')
        torch.save(interpretation, out_dir / 'interpretations.pkl')
        return results, targets, interpretation

    def get_interpretation_callback(self, **kwargs):
        tensorboard_converters = TensorboardConverters(
            logdir=kwargs['logdir'],
            tensorboard_loggers=self.tensor_loggers,
            datasets=kwargs.get('datasets', self.get_default_datasets(**kwargs))
        )
        interpretation_callback = InterpretationCallback(self.task_flow, tensorboard_converters)
        return interpretation_callback

    def get_default_loaders(self, shuffle_train=True):
        datasets = self.get_default_datasets()
        train_dataset = datasets['train']
        val_dataset = datasets['valid']
        test_dataset = datasets['test']
        train_loader = DataLoader(train_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=False)
        loaders = OrderedDict({
            'train': train_loader,
            'valid': val_loader,
            'test': test_loader
        })

        # Rename 'train' loader and dataset, since catalyst does not allow inference on train dataset.
        if not shuffle_train:
            loaders['infer'] = loaders['train']
            del loaders['train']
            datasets['infer'] = datasets['train']
            del datasets['train']
        return datasets, loaders

    def get_default_datasets(self, **kwargs):
        dataset = self.task_flow.get_dataset()
        if self.train_test_val_indices is None:
            raise ValueError(f'You must supply either a `loaders` parameter, or give `train_test_val_indices` via'
                             f'constructor.')
        train_indices, test_indices, val_indices = self.train_test_val_indices
        train_dataset = TransformedSubset(dataset, train_indices)
        val_dataset = TransformedSubset(dataset, val_indices)
        test_dataset = TransformedSubset(dataset, test_indices)

        datasets = {
            'train': train_dataset,
            'valid': val_dataset,
            'test': test_dataset,
        }

        datasets['infer'] = datasets[kwargs.get('target_loader', 'valid')]
        return datasets

    def batch_to_device(self, batch, device):
        return super()._batch2device(batch, device)

    def batch_to_model_device(self, batch, model):
        return super()._batch2device(batch, next(model.parameters()).device)
