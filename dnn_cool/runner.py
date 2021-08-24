from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial, cached_property
from pathlib import Path
from shutil import copyfile
from time import time
from typing import Dict, Tuple, Callable, Optional, Any, Union, List, Sized
from typing import Iterator, Mapping

import numpy as np
import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, InferCallback, State, Callback
from catalyst.utils import load_checkpoint, unpack_checkpoint, any2device
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters, ReplaceGatherCallback, \
    TensorboardConverter
from dnn_cool.tasks import TaskFlow, TaskFlowForDevelopment
from dnn_cool.utils import TransformedSubset, train_test_val_split, load_model_from_export


@dataclass
class TrainingArguments(Mapping):
    num_epochs: int
    criterion: Optional[Callable] = None
    model: Optional[Callable] = None
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[Any] = None
    logdir: Optional[Union[str, Path]] = None
    loaders: Optional[Dict[str, DataLoader]] = None
    callbacks: Optional[Union[List[Callback], Dict[str, Callback]]] = None
    fp16: Union[Dict, bool] = None
    catalyst_args: Dict = field(default_factory=lambda: {})
    train_transforms: Callable = None
    val_transforms: Callable = None

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

    def __init__(self, out_key='logits', loaders_to_skip=(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaders_to_skip = loaders_to_skip
        self.out_key = out_key
        self.predictions = {}
        self.targets = {}
        self.__current_store = None

    def on_loader_start(self, state: State):
        self.predictions[state.loader_key] = {}
        self.targets[state.loader_key] = {}

    def on_batch_end(self, state: State):
        dct = state.output[self.out_key]
        if '_device|overall|_n' in dct:
            if self.__current_store is not None:
                self.__current_store(loader_name=state.loader_key)
            return
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items() if key != 'gt'}
        loader_name = state.loader_key
        targets = state.input['targets']
        self.update_storage(loader_name, dct, targets)

    def on_dataparallel_gather(self, dct):
        self.__current_store = partial(self.update_storage, dct=dct, targets=dct['gt']['_targets'])

    def update_storage(self, loader_name, dct, targets):
        for key, value in dct.items():
            if key == 'gt':
                continue
            if key not in self.predictions[loader_name]:
                self.predictions[loader_name][key] = []
            if isinstance(value, list):
                self.predictions[loader_name][key].extend(value)
            else:
                self.predictions[loader_name][key].append(value)
        for key, value in targets.items():
            if key == 'gt':
                continue
            if key not in self.targets[loader_name]:
                self.targets[loader_name][key] = []
            if isinstance(value, list):
                self.targets[loader_name][key].extend(value)
            else:
                self.targets[loader_name][key].append(value.detach().cpu().numpy())

    def on_loader_end(self, state: State):
        self.predictions[state.loader_key] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions[state.loader_key].items()
        }

        self.targets[state.loader_key] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.targets[state.loader_key].items()
        }


def load_inference_results_from_directory(logdir):
    out_dir = logdir / 'infer'
    out_dir.mkdir(exist_ok=True)
    res = {}
    for key in ['logits', 'targets', 'interpretations']:
        file_path = out_dir / f'{key}.pkl'
        if file_path.exists():
            res[key] = torch.load(out_dir / f'{key}.pkl')
    return res


class DnnCoolRunnerView:

    def __init__(self, full_flow: TaskFlow, model: nn.Module,
                 project_dir: Union[str, Path], runner_name: str):
        self.project_dir = Path(project_dir)
        self.full_flow = full_flow
        self.model = model
        self.runner_name = runner_name
        self.default_logdir_name = f'./logdir_{runner_name}'

    def best(self) -> nn.Module:
        return self.load_model_from_checkpoint('best')

    def last(self) -> nn.Module:
        return self.load_model_from_checkpoint('last')

    def from_epoch(self, i) -> nn.Module:
        return self.load_model_from_checkpoint(f'train.{i}')

    def load_model_from_export(self, out_directory: Union[str, Path]) -> nn.Module:
        return load_model_from_export(self.model, self.full_flow, out_directory)

    def load_model_from_checkpoint(self, checkpoint_name) -> nn.Module:
        model = self.model
        logdir = self.project_dir / self.default_logdir_name
        checkpoint_path = str(logdir / 'checkpoints' / f'{checkpoint_name}.pth')
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        thresholds_path = logdir / 'tuned_params.pkl'
        if not thresholds_path.exists():
            return model
        tuned_params = torch.load(thresholds_path)
        self.full_flow.get_decoder().load_tuned(tuned_params)
        return model

    def load_tuned_params(self) -> Optional[Dict]:
        logdir = self.project_dir / self.default_logdir_name
        thresholds_path = logdir / 'tuned_params.pkl'
        if not thresholds_path.exists():
            return None
        return torch.load(thresholds_path)

    def load_inference_results(self) -> Dict:
        logdir = self.project_dir / self.default_logdir_name
        return load_inference_results_from_directory(logdir)

    def load_train_test_val_split(self):
        return read_split(self.project_dir / self.default_logdir_name)

    @cached_property
    def train_test_val_indices(self):
        return self.load_train_test_val_split()

    @cached_property
    def inference_results(self):
        return self.load_inference_results()

    @cached_property
    def evaluation_df(self):
        import pandas as pd
        return pd.read_csv(self.project_dir / self.default_logdir_name / 'evaluation.csv')

    def summarize_loss_values(self, loader_name, task_name):
        interpretations = self.inference_results['interpretations'][loader_name]
        loss_values = interpretations[task_name]
        loss_local_indices = interpretations[f'indices|{task_name}']
        loader_idx = ['infer', 'test', 'valid'].index(loader_name)
        global_loader_indices = self.train_test_val_indices[loader_idx]
        mask = loss_local_indices >= 0
        logits = self.inference_results['logits'][loader_name][task_name][loss_local_indices[mask]]
        task = self.full_flow.get_all_children()[task_name]
        logits = torch.tensor(logits)
        activated = task.get_activation()(logits) if task.get_activation() is not None else logits
        decoded = task.get_decoder()(activated) if task.get_decoder() is not None else activated
        return {
            'global_idx': global_loader_indices[loss_local_indices[mask]],
            'loss_values': loss_values[mask],
            'targets': self.inference_results['targets'][loader_name][task_name][loss_local_indices[mask]],
            'activated': activated,
            'decoded': decoded.numpy(),
            'logits': logits.numpy(),
            'task': task
        }

    def worst_examples(self, loader_name, task_name, n):
        return self.extremal_examples(loader_name, task_name, n, -1)

    def best_examples(self, loader_name, task_name, n):
        return self.extremal_examples(loader_name, task_name, n, 1)

    def extremal_examples(self, loader_name, task_name, n, mult):
        res = self.summarize_loss_values(loader_name, task_name)
        sorter = (mult * res['loss_values']).argsort()
        top_n_idx = res['global_idx'][sorter[:n]]
        return top_n_idx, res


def batch_to_device(batch, device) -> Mapping[str, torch.Tensor]:
    return any2device(batch, device)


class DnnCoolSupervisedRunner(SupervisedRunner):

    def __init__(self, model: nn.Module,
                 full_flow: TaskFlowForDevelopment,
                 project_dir: Union[str, Path],
                 runner_name: str,
                 tensoboard_converters: TensorboardConverter,
                 early_stop: bool = True,
                 balance_dataparallel_memory: bool = False,
                 train_test_val_indices: Tuple[np.ndarray, np.ndarray, np.ndarray] = None):
        self.task_flow = full_flow

        self.default_criterion = self.task_flow.get_criterion()
        self.balance_dataparallel_memory = balance_dataparallel_memory

        self.default_callbacks = []
        if self.balance_dataparallel_memory:
            self.default_callbacks.append(ReplaceGatherCallback(self.task_flow))
        self.default_callbacks.extend(self.default_criterion.catalyst_callbacks())
        self.default_optimizer = partial(optim.AdamW, lr=1e-4)
        self.default_scheduler = ReduceLROnPlateau
        self.project_dir: Path = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
        self.default_logdir = f'./logdir_{runner_name}'

        if early_stop:
            self.default_callbacks.append(EarlyStoppingCallback(patience=5))

        (self.project_dir / self.default_logdir).mkdir(exist_ok=True)
        if train_test_val_indices is None:
            n = len(self.task_flow.get_dataset())
            train_test_val_indices = runner_split(n, self.project_dir / self.default_logdir)
        else:
            save_split(self.project_dir / self.default_logdir, train_test_val_indices)
        self.train_test_val_indices = train_test_val_indices
        self.tensor_loggers = tensoboard_converters
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

        kwargs['logdir'] = kwargs.get('logdir', self.default_logdir)
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
        infer_dict_callback = InferDictCallback()
        default_callbacks = OrderedDict([("interpretation", interpretation_callback),
                                         ("inference", infer_dict_callback)])
        if self.balance_dataparallel_memory:
            replace_gather_callback = ReplaceGatherCallback(self.task_flow, infer_dict_callback)
            default_callbacks["dataparallel_reducer"] = replace_gather_callback
        kwargs['callbacks'] = kwargs.get('callbacks', default_callbacks)
        kwargs['model'] = kwargs.get('model', self.model)
        store = kwargs.pop('store', True)
        kwargs.pop('loader_names_to_skip_in_interpretation', ())
        del kwargs['datasets']
        super().infer(*args, **kwargs)
        res = {}
        if 'inference' in kwargs['callbacks']:
            res['logits'] = kwargs['callbacks']['inference'].predictions
            res['targets'] = kwargs['callbacks']['inference'].targets
        if 'interpretation' in kwargs['callbacks']:
            res['interpretations'] = kwargs['callbacks']['interpretation'].interpretations

        if store:
            out_dir = logdir / 'infer'
            out_dir.mkdir(exist_ok=True)
            for key in res:
                torch.save(res[key], out_dir / f'{key}.pkl', pickle_protocol=4)
        return res

    def create_interpretation_callback(self, **kwargs) -> InterpretationCallback:
        tensorboard_converters = TensorboardConverters(
            logdir=kwargs['logdir'],
            tensorboard_loggers=self.tensor_loggers,
            datasets=kwargs.get('datasets', self.get_default_datasets(**kwargs))
        )
        loaders_to_skip = kwargs.get('loader_names_to_skip_in_interpretation', ())
        interpretation_callback = InterpretationCallback(self.task_flow.get_per_sample_criterion(),
                                                         tensorboard_converters,
                                                         loaders_to_skip)
        return interpretation_callback

    def get_default_loaders(self, shuffle_train=True,
                            collator=None,
                            batch_size_per_gpu=32) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
        datasets = self.get_default_datasets()
        train_dataset = datasets['train']
        val_dataset = datasets['valid']
        test_dataset = datasets['test']
        bs = max(batch_size_per_gpu, batch_size_per_gpu * torch.cuda.device_count())
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=shuffle_train, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collator)
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
        train_dataset = TransformedSubset(dataset, train_indices, sample_transforms=kwargs.get('train_transforms'))
        val_dataset = TransformedSubset(dataset, val_indices, sample_transforms=kwargs.get('val_transforms'))
        test_dataset = TransformedSubset(dataset, test_indices, sample_transforms=kwargs.get('val_transforms'))

        datasets = {
            'train': train_dataset,
            'valid': val_dataset,
            'test': test_dataset,
        }

        datasets['infer'] = datasets[kwargs.get('target_loader', 'valid')]
        return datasets

    def batch_to_model_device(self, batch) -> Mapping[str, torch.Tensor]:
        return any2device(batch, next(self.model.parameters()).device)

    def best(self) -> nn.Module:
        model = self.model
        checkpoint_path = str(self.project_dir / self.default_logdir / 'checkpoints' / 'best_full.pth')
        ckpt = load_checkpoint(checkpoint_path)
        unpack_checkpoint(ckpt, model)

        thresholds_path = self.project_dir / self.default_logdir / 'tuned_params.pkl'
        if not thresholds_path.exists():
            return model
        tuned_params = torch.load(thresholds_path)
        self.task_flow.task.get_decoder().load_tuned(tuned_params)
        return model

    def tune(self, loader_name='valid', store=True) -> Dict:
        res = self.load_inference_results()
        decoder = self.task_flow.task.get_decoder()
        tuned_params = decoder.tune(res['logits'][loader_name], res['targets'][loader_name])
        if store:
            out_path = self.project_dir / self.default_logdir / 'tuned_params.pkl'
            torch.save(tuned_params, out_path)
        return tuned_params

    def load_inference_results(self) -> Dict:
        logdir = self.project_dir / self.default_logdir
        return load_inference_results_from_directory(logdir)

    def load_tuned(self) -> Dict:
        tuned_params = torch.load(self.project_dir / self.default_logdir / 'tuned_params.pkl')
        self.task_flow.task.get_decoder().load_tuned(tuned_params)
        return tuned_params

    def evaluate(self, loader_name='test'):
        res = self.load_inference_results()
        self.load_tuned()
        evaluator = self.task_flow.get_evaluator()
        df = evaluator(res['logits'][loader_name], res['targets'][loader_name])
        df.to_csv(self.project_dir / self.default_logdir / 'evaluation.csv', index=False)
        return df

    def export_for_deployment(self, out_directory: Path):
        params_file = self.project_dir / self.default_logdir / 'tuned_params.pkl'
        if params_file.exists():
            copyfile(params_file, out_directory / 'tuned_params.pkl')
        torch.save(self.model.state_dict(), out_directory / 'state_dict.pth')


def split_already_done(n: int, project_dir):
    total_len = 0
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = project_dir / f'{split_name}_indices.npy'
        if not split_path.exists():
            return False
        total_len += len(np.load(split_path))

    return total_len == n


def read_split(runner_dir):
    res = []
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = runner_dir / f'{split_name}_indices.npy'
        res.append(np.load(split_path))
    return res


def save_split(project_dir, res):
    for i, split_name in enumerate(['train', 'test', 'val']):
        split_path = project_dir / f'{split_name}_indices.npy'
        np.save(split_path, res[i])


def runner_split(n: int, runner_dir):
    if split_already_done(n, runner_dir):
        return read_split(runner_dir)
    res = train_test_val_split(n)
    save_split(runner_dir, res)
    return res
