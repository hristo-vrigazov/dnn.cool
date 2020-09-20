from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from time import time
from typing import Dict, Tuple, Callable, Optional, Any, Union, List
from torch import nn

import numpy as np
import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, InferCallback, State, Callback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters
from dnn_cool.utils import TransformedSubset, train_test_val_split


@dataclass
class TrainingArguments(Mapping):
    num_epochs: int
    criterion: Optional[Callable] = None
    model: Optional[Callable] = None
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[Any] = None
    logdir: Optional[Union[str, Path]] = None
    loaders: Optional[OrderedDictType[str, DataLoader]] = None
    callbacks: Optional[Union[List[Callback], OrderedDictType[str, Callback]]] = None
    fp16: Union[Dict, bool] = None
    catalyst_args: Dict = field(default_factory=lambda: {})

    def __getitem__(self, k):
        if hasattr(self, k):
            return getattr(self, k)
        return self.catalyst_args[k]

    def __len__(self) -> int:
        return len(self.__dataclass_fields__) + len(self.catalyst_args)

    def __iter__(self) -> Iterator:
        res = {}
        for field_name in self.__dataclass_fields__:
            if field_name != 'catalyst_args':
                attr = getattr(self, field_name)
                if attr is not None:
                    res[field_name] = attr
        for key, value in self.catalyst_args.items():
            res[key] = value
        return iter(res)


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
        dct = state.output[self.out_key]
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            if key not in self.predictions[state.loader_name]:
                self.predictions[state.loader_name][key] = []
            self.predictions[state.loader_name][key].append(value)

        targets = state.input['targets']
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

    def __init__(self, project, model, early_stop: bool = True, runner_name=None, train_test_val_indices=None):
        self.task_flow = project.get_full_flow()

        self.default_criterion = self.task_flow.get_loss()
        self.default_callbacks = self.default_criterion.catalyst_callbacks()
        self.default_optimizer = partial(optim.AdamW, lr=1e-4)
        self.default_scheduler = ReduceLROnPlateau
        self.project_dir: Path = project.project_dir
        self.project_dir.mkdir(exist_ok=True)
        runner_name = f'{self.task_flow.get_name()}_{time()}' if runner_name is None else runner_name
        self.default_logdir = f'./logdir_{runner_name}'

        if early_stop:
            self.default_callbacks.append(EarlyStoppingCallback(patience=5))

        if train_test_val_indices is None:
            (self.project_dir / self.default_logdir).mkdir(exist_ok=True)
            train_test_val_indices = project_split(project.df, self.project_dir / self.default_logdir)
        else:
            save_split(self.project_dir / self.default_logdir, train_test_val_indices)
        self.train_test_val_indices = train_test_val_indices
        self.tensor_loggers = project.converters.tensorboard_converters
        super().__init__(model=model)

    def train(self, *args, **kwargs):
        kwargs['criterion'] = kwargs.get('criterion', self.default_criterion)
        kwargs['model'] = kwargs.get('model', self.model)

        if 'optimizer' not in kwargs:
            model = kwargs['model']
            optimizable_params = filter(lambda p: p.requires_grad, model.parameters())
            kwargs['optimizer'] = self.default_optimizer(params=optimizable_params)

        if 'scheduler' not in kwargs:
            kwargs['scheduler'] = self.default_scheduler(kwargs['optimizer'])

        if 'logdir' not in kwargs:
            kwargs['logdir'] = self.default_logdir
        kwargs['logdir'] = self.project_dir / kwargs['logdir']
        kwargs['num_epochs'] = kwargs.get('num_epochs', 50)

        if 'loaders' not in kwargs:
            datasets, kwargs['loaders'] = self.get_default_loaders()

        default_callbacks = [self.create_interpretation_callback(**kwargs)] + self.default_callbacks
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)
        super().train(*args, **kwargs)

    def infer(self, *args, **kwargs):
        default_datasets, default_loaders = self.get_default_loaders(shuffle_train=False)
        kwargs['loaders'] = kwargs.get('loaders', default_loaders)
        kwargs['datasets'] = kwargs.get('datasets', default_datasets)

        logdir = self.project_dir / Path(kwargs.get('logdir', self.default_logdir))
        kwargs['logdir'] = logdir
        interpretation_callback = self.create_interpretation_callback(**kwargs)
        default_callbacks = OrderedDict([("interpretation", interpretation_callback),
                                         ("inference", InferDictCallback())])
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)
        kwargs['model'] = kwargs.get('model', self.model)
        del kwargs['datasets']
        super().infer(*args, **kwargs)
        results = kwargs['callbacks']['inference'].predictions
        targets = kwargs['callbacks']['inference'].targets
        interpretation = kwargs['callbacks']['interpretation'].interpretations

        out_dir = logdir / 'infer'
        out_dir.mkdir(exist_ok=True)
        torch.save(results, out_dir / 'logits.pkl')
        torch.save(targets, out_dir / 'targets.pkl')
        torch.save(interpretation, out_dir / 'interpretations.pkl')
        return results, targets, interpretation

    def create_interpretation_callback(self, **kwargs) -> InterpretationCallback:
        tensorboard_converters = TensorboardConverters(
            logdir=kwargs['logdir'],
            tensorboard_loggers=self.tensor_loggers,
            datasets=kwargs.get('datasets', self.get_default_datasets(**kwargs))
        )
        interpretation_callback = InterpretationCallback(self.task_flow, tensorboard_converters)
        return interpretation_callback

    def get_default_loaders(self, shuffle_train=True,
                            collator=None,
                            batch_size_per_gpu=32) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
        datasets = self.get_default_datasets()
        train_dataset = datasets['train']
        val_dataset = datasets['valid']
        test_dataset = datasets['test']
        train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu * torch.cuda.device_count(), shuffle=shuffle_train,
                                  collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu * torch.cuda.device_count(), shuffle=False,
                                collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_per_gpu * torch.cuda.device_count(), shuffle=False,
                                 collate_fn=collator)
        loaders = OrderedDict({
            'train': train_loader,
            'valid': val_loader,
        })

        # Rename 'train' loader and dataset, since catalyst does not allow inference on train dataset.
        if not shuffle_train:
            loaders['infer'] = loaders['train']
            del loaders['train']
            datasets['infer'] = datasets['train']
            del datasets['train']
            loaders['test'] = test_loader
        return datasets, loaders

    def get_default_datasets(self, **kwargs) -> Dict[str, Dataset]:
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

    def batch_to_device(self, batch, device) -> Mapping[str, torch.Tensor]:
        return super()._batch2device(batch, device)

    def batch_to_model_device(self, batch) -> Mapping[str, torch.Tensor]:
        return super()._batch2device(batch, next(self.model.parameters()).device)

    def best(self) -> nn.Module:
        model = self.model
        checkpoint_path = self.project_dir / self.default_logdir / 'checkpoints' / 'best_full.pth'
        ckpt = load_checkpoint(checkpoint_path)
        unpack_checkpoint(ckpt, model)

        thresholds_path = self.project_dir / self.default_logdir / 'tuned_params.pkl'
        if not thresholds_path.exists():
            return model
        tuned_params = torch.load(thresholds_path)
        self.task_flow.get_decoder().load_tuned(tuned_params)
        return model

    def tune(self) -> Dict:
        predictions, targets, interpretations = self.load_inference_results()
        decoder = self.task_flow.get_decoder()
        tuned_params = decoder.tune(predictions['valid'], targets['valid'])
        out_path = self.project_dir / self.default_logdir / 'tuned_params.pkl'
        torch.save(tuned_params, out_path)
        return tuned_params

    def load_inference_results(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        logdir = self.project_dir / self.default_logdir
        out_dir = logdir / 'infer'
        out_dir.mkdir(exist_ok=True)
        results = torch.load(out_dir / 'logits.pkl')
        targets = torch.load(out_dir / 'targets.pkl')
        interpretation = torch.load(out_dir / 'interpretations.pkl')
        return results, targets, interpretation

    def load_tuned(self) -> Dict:
        tuned_params = torch.load(self.project_dir / self.default_logdir / 'tuned_params.pkl')
        self.task_flow.get_decoder().load_tuned(tuned_params)
        return tuned_params

    def evaluate(self):
        predictions, targets, interpretations = self.load_inference_results()
        self.load_tuned()
        evaluator = self.task_flow.get_evaluator()
        df = evaluator(predictions['test'], targets['test'])
        df.to_csv(self.project_dir / self.default_logdir / 'evaluation.csv', index=False)
        return df


def split_already_done(df, project_dir):
    total_len = 0
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = project_dir / f'{split_name}_indices.npy'
        if not split_path.exists():
            return False
        total_len += len(np.load(split_path))

    if total_len != len(df):
        return False
    return True


def read_split(project_dir):
    res = []
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = project_dir / f'{split_name}_indices.npy'
        res.append(np.load(split_path))
    return res


def save_split(project_dir, res):
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = project_dir / f'{split_name}_indices.npy'
        np.save(split_path, res[i])


def project_split(df, project_dir):
    if split_already_done(df, project_dir):
        return read_split(project_dir)
    res = train_test_val_split(df)
    save_split(project_dir, res)
    return res
