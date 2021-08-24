from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Container, Sequence
from typing import Callable, Dict, Optional, Tuple, List, Mapping

import numpy as np
import torch
from catalyst.callbacks import BatchMetricCallback
from catalyst.callbacks import IMetricCallback
from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core import Callback, CallbackOrder, State, IRunner
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, SequentialSampler

from dnn_cool.utils import any_value


def publish_all(prefix: str,
                idx: int,
                sample: Tuple[Dict, Dict],
                mapping_key: str,
                key: str,
                writer: SummaryWriter,
                mapping: Mapping,
                task_name: str):
    """
    Publishes a given key given all publishers supplied via the mapping
    :param prefix: The prefix, this is typically either "best" or "worst".

    :param idx: The index of the sample in the dataset.

    :param sample: A tuple X, y where X and y and dictionaries.

    :param mapping_key: The key which when applied to the mapping gives a list of publishers.

    :param key: The key in X that is being published.

    :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

    :param mapping: A mapping `[str, List[Callable]]` where the key are either input columns, or input types and the values are a list of publisher functions.

    :param task_name: The name of the task, to be included in the name
    """
    if mapping_key in mapping:
        publishers: List[ITensorboardPublisher] = mapping[mapping_key]
        for publisher in publishers:
            X, y = sample
            tag = f'{prefix}_{task_name}'
            publisher(writer, tag, X[key], idx)


class ITensorboardPublisher:

    def __call__(self, writer: SummaryWriter, tag: str, sample: Any, idx: int):
        """
        Publishes interpretation data.

        :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

        :param tag: The prefix, this is typically either "best_key" or "worst_key".

        :param sample: The sample that is going to be published to the Tensorboard

        :param idx: The index of the dataset sample.

        """
        raise NotImplementedError()


def img_publisher(writer: SummaryWriter, tag: str, sample: Any, idx: int):
    writer.add_image(f'{tag}_images', sample, global_step=idx)


def text_publisher(writer: SummaryWriter, tag: str, sample: Any, idx: int):
    writer.add_text(f'{tag}_text', sample, global_step=idx)


def default_tensorboard_type_mapping():
    return {
        'img': [img_publisher],
        'text': [text_publisher]
    }


@dataclass()
class TensorboardConverter:
    """
    A dataclass which holds mappings from column names to Tensorboard publishers and from column types to Tensorboard
    publishers. Also, it stores which column is of what type, to be able to log any column name to the Tensorboard.
    """
    col_mapping: Dict[str, List[ITensorboardPublisher]] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column names to a list of publishers. A publisher is just a callable which will be called 
    with this signature: 
    :code:`publisher(writer: SummaryWriter, idx: int, sample: Tuple, prefix: str, task_name: str, key: str)`. Example 
    publisher functions are :meth:`dnn_cool.catalyst_utils.img` and :meth:`dnn_cool.catalyst_utils.text`.
    """
    type_mapping: Dict[str, List[ITensorboardPublisher]] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column types to a list of publishers. A publisher is just a callable which will be called 
    with this signature: 
    :code:`publisher(writer: SummaryWriter, idx: int, sample: Tuple, prefix: str, task_name: str, key: str)`. Example 
    publisher functions are :meth:`dnn_cool.catalyst_utils.img` and :meth:`dnn_cool.catalyst_utils.text`.
    """
    col_to_type_mapping: Dict[str, str] = field(default_factory=lambda: {})
    """
    Stores a `dict` from column names to type names.
    """

    def __call__(self, writer: SummaryWriter, idx: int, sample: Tuple[Dict, Dict], prefix: str, task_name: str):
        """
        Publishes a given sample to the tensorboard, using the mappings defined in the dataclass.

        :param writer: A :class:`SummaryWriter` that logs to the tensorboard.

        :param idx: The index of the sample that is being published

        :param sample: A tuple of dictionaries X, y.

        :param prefix: The prefix, this is typically either "best" or "worst".

        :param task_name: The name of the task

        :return:
        """
        if task_name == 'gt':
            return
        X, y = sample
        for key in X:
            publish_all(prefix, idx, sample, key, key, writer, self.col_mapping, task_name)
        for key in X:
            if key in self.col_to_type_mapping:
                publish_all(prefix,
                            idx,
                            sample,
                            self.col_to_type_mapping[key],
                            key,
                            writer,
                            self.type_mapping,
                            task_name)


@dataclass
class TensorboardConverters:
    """
    This class handles the logging to the Tensorboard of an interpretation for a task
    """
    logdir: Path
    datasets: Dict[str, Dataset]
    tensorboard_loggers: Callable = field(default_factory=TensorboardConverter)
    loggers: Dict[str, SummaryWriter] = field(default_factory=lambda: {})
    top_k: int = 10

    def initialize(self, state: State):
        """
        Initializes the tensorboard loggers.

        :param state: The state with which the callback is called.
        """
        if (self.logdir is not None) and (state.loader_key not in self.loggers):
            path = str(self.logdir / f"{state.loader_key}_log")
            writer = SummaryWriter(path)
            self.loggers[state.loader_key] = writer

    def publish(self, state: State, interpretations: Dict[str, torch.Tensor]):
        """
        Publishes all interpretations

        :param state: The state with which the callback is called.

        :param interpretations: A dict object from task name to loss values. Also additional keys are those prefixed \
        with "indices|{task_name}", which hold the corresponding indices in the original dataset for which the loss \
        items are computed.

        """
        for key, value in interpretations.items():
            if key.startswith('indices'):
                continue
            sorted_indices = value.argsort()
            best_indices = interpretations[f'indices|{key}'][sorted_indices][:self.top_k]
            worst_indices = interpretations[f'indices|{key}'][sorted_indices][-self.top_k:]
            writer: SummaryWriter = self.loggers[state.loader_key]
            dataset = self.datasets[state.loader_key]
            self._publish_inputs(best_indices, writer, dataset, prefix='best', key=key)
            self._publish_inputs(worst_indices, writer, dataset, prefix='worst', key=key)

    def _publish_inputs(self, best_indices, writer, dataset, prefix, key):
        for idx in best_indices:
            if self.tensorboard_loggers is not None:
                self.tensorboard_loggers(writer, idx, dataset[idx], prefix, key)

    def close(self, state: State):
        """Close opened tensorboard writers"""
        if state.logdir is None:
            return

        for logger in self.loggers.values():
            logger.close()


def should_skip_loader(state, loaders_to_skip):
    if not isinstance(state.loaders[state.loader_name].sampler, SequentialSampler):
        return True
    if state.loader_name in loaders_to_skip:
        return True
    return False


class InterpretationCallback(Callback):
    """
    This callback publishes best and worst images per task, according to the configuration supplied via the constructor.
    """

    def __init__(self, per_sample_criterion,
                 tensorboard_converters: Optional[TensorboardConverters] = None,
                 loaders_to_skip=()):
        """
        :param flow: The task flow, which holds the per sample loss functions for every task.

        :param tensorboard_converters: A :class:`TensorboardConverters` object which is responsible for the Tensorboard
        logging settings.

        :param loaders_to_skip: Optional loaders to be skipped, for example because labels aren't available for them.
        """
        super().__init__(CallbackOrder.Metric)
        self.loaders_to_skip = loaders_to_skip

        self.overall_loss = per_sample_criterion
        self.leaf_losses = self.overall_loss.get_leaf_losses_per_sample()
        self.interpretations = {}
        self.loader_counts = {}

        self.tensorboard_converters = tensorboard_converters

    def _initialize_interpretations(self):
        interpretation_dict = {
            'overall': [],
            'indices|overall': []
        }
        for path in self.leaf_losses:
            interpretation_dict[path] = []
            interpretation_dict[f'indices|{path}'] = []
        return interpretation_dict

    def on_loader_start(self, state: State):
        if should_skip_loader(state, self.loaders_to_skip):
            return
        self.interpretations[state.loader_name] = self._initialize_interpretations()
        self.loader_counts[state.loader_name] = 0

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.initialize(state)

    def on_batch_end(self, state: State):
        if should_skip_loader(state, self.loaders_to_skip):
            return
        outputs = state.output['logits']
        targets = state.input['targets']
        overall_res = self.overall_loss(outputs, targets)
        start = self.loader_counts[state.loader_name]

        n = 0
        for path, loss in overall_res.items():
            if path.startswith('indices'):
                continue
            self.interpretations[state.loader_name][path].append(loss.detach().cpu().numpy())
            ind_key = f'indices|{path}'
            indices = overall_res[ind_key] + start
            self.interpretations[state.loader_name][ind_key].append(indices.detach().cpu().numpy())
            n = len(indices)

        self.loader_counts[state.loader_name] += n

    def on_loader_end(self, state: State):
        if should_skip_loader(state, self.loaders_to_skip):
            return
        self.interpretations[state.loader_name] = self.prepare_interpretations(state)

        if self.tensorboard_converters is not None:
            self.tensorboard_converters.publish(state, self.interpretations[state.loader_name])

    def prepare_interpretations(self, state):
        res = {}
        for key, value in self.interpretations[state.loader_name].items():
            arrs = []
            for arr in value:
                try:
                    if len(arr) > 0:
                        arrs.append(arr)
                except TypeError:
                    pass
            value = np.concatenate(arrs)
            res[key] = value
        return res

    def on_stage_end(self, state: State):
        if should_skip_loader(state, self.loaders_to_skip):
            return
        if self.tensorboard_converters is not None:
            self.tensorboard_converters.close(state)


class DeviceReducingDataParallel(DataParallel):

    def __init__(self, module: nn.Module, task_flow, infer_dict_callback):
        super().__init__(module)
        tasks_dict = task_flow.get_all_children()
        self.full_paths = []
        for full_path, task_for_development in tasks_dict.items():
            if task_for_development.task.is_train_only():
                continue
            self.full_paths.append(full_path)
        criterion = task_flow.get_criterion()
        per_sample_criterion = task_flow.get_per_sample_criterion(ctx=criterion.get_device_reduced_ctx())
        leaf_losses = criterion.get_leaf_losses()
        metrics = criterion.get_metrics()
        self._reducing_func = partial(reduce_on_device,
                                      criterion=criterion,
                                      per_sample_criterion=per_sample_criterion,
                                      leaf_criterions=leaf_losses,
                                      metrics=metrics)
        self.infer_dict_callback = infer_dict_callback
        self.ctx = criterion.ctx
        self.r_device_metrics = False
        self.r_leaf_losses = False
        self.r_per_sample_losses = False
        self.store_inference_results = self.infer_dict_callback is not None

    def gather(self, outputs, output_device):
        self.ctx.clear()
        device_reduced_results = []
        dct = {
            'gt': {'_targets': {}}
        }
        for full_path in self.full_paths:
            dct[full_path] = []
            dct[f'precondition|{full_path}'] = []

        ctx_reductions = {}
        for i in range(len(outputs)):
            reduced_with_grad, reduced = self._reducing_func(outputs=outputs[i],
                                                             targets=outputs[i]['gt']['_targets'],
                                                             r_device_metrics=self.r_device_metrics,
                                                             r_leaf_losses=self.r_leaf_losses,
                                                             r_per_sample_losses=self.r_per_sample_losses)
            device_reduced_results.append(reduced_with_grad)
            for key, value in reduced.items():
                if key not in ctx_reductions:
                    ctx_reductions[key] = []
                value = value.detach().cpu()
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                ctx_reductions[key].append(value)

            if self.store_inference_results:
                for full_path in self.full_paths:
                    dct[full_path].append(outputs[i][full_path].detach().cpu().numpy())
                    precondition_path = f'precondition|{full_path}'
                    dct[precondition_path].append(outputs[i][precondition_path].detach().cpu().numpy())
                    np_targets = outputs[i]['gt']['_targets'][full_path].detach().cpu().numpy()
                    if full_path not in dct['gt']['_targets']:
                        dct['gt']['_targets'][full_path] = []
                    dct['gt']['_targets'][full_path].append(np_targets)

        if self.infer_dict_callback is not None:
            self.infer_dict_callback.on_dataparallel_gather(dct)
        gathered = super().gather(device_reduced_results, output_device)
        additional_metrics = {key: torch.cat(value, dim=0) for key, value in ctx_reductions.items()}

        for key, value in additional_metrics.items():
            self.ctx[key] = value
        return gathered

    def reset_device_reducing_tasks(self):
        self.r_device_metrics = False
        self.r_leaf_losses = False
        self.r_per_sample_losses = False
        self.store_inference_results = len(self.callbacks) > 0


class ReplaceGatherCallback(Callback):

    def __init__(self, task_flow, infer_dict_callback=None):
        super().__init__(CallbackOrder.External)
        self.task_flow = task_flow
        self.infer_dict_callback = infer_dict_callback

    def on_stage_start(self, runner: "IRunner"):
        if isinstance(runner.model, DataParallel):
            runner.model = DeviceReducingDataParallel(runner.model.module, self.task_flow, self.infer_dict_callback)

    def on_loader_start(self, runner: "IRunner"):
        model = runner.model
        if not isinstance(model, DeviceReducingDataParallel):
            return
        for idx, callback in runner.callbacks.items():
            if isinstance(callback, InterpretationCallback):
                model.r_per_sample_losses = not should_skip_loader(runner, callback.loaders_to_skip)
            if isinstance(callback, BatchMetricCallback):
                model.r_device_metrics = True
                model.r_leaf_losses = True


def reduce_on_device(criterion,
                     per_sample_criterion,
                     leaf_criterions,
                     metrics,
                     outputs,
                     targets,
                     r_device_metrics,
                     r_leaf_losses,
                     r_per_sample_losses):
    loss = criterion(outputs, targets)
    any_tensor = any_value(targets)
    n = len(any_tensor)
    criterion_n = torch.tensor(n, dtype=any_tensor.dtype, device=any_tensor.device)
    reduced_with_grad = {
        f'_device|{criterion.prefix}{criterion.task_name}|loss': loss,
        f'_device|{criterion.prefix}{criterion.task_name}|_n': criterion_n,
        f'_device|overall|loss': loss,
        f'_device|overall|_n': criterion_n
    }
    reduced = {}

    with torch.no_grad():
        if r_device_metrics:
            compute_device_metrics(reduced, any_tensor, metrics, outputs, targets)
        if r_leaf_losses:
            compute_leaf_losses(leaf_criterions, outputs, reduced, targets)
        if r_per_sample_losses:
            compute_per_sample_losses(reduced, per_sample_criterion, outputs, targets, n)

    return reduced_with_grad, reduced


def compute_leaf_losses(leaf_criterions, outputs, reduced, targets):
    for path, leaf_loss in leaf_criterions.items():
        reduced[f'_device|{path}|loss'] = leaf_loss(outputs, targets).loss_items
        reduced[f'_device|{path}|_n'] = outputs[f'precondition|{path}'].sum()


def compute_per_sample_losses(reduced, per_sample_criterion, outputs, targets, n):
    per_sample_losses = per_sample_criterion(outputs, targets)
    for key, value in per_sample_losses.items():
        if key.startswith('indices'):
            value += (n * value.device.index)
        if len(value.shape) == 0:
            value = value.unsqueeze(0)
        reduced[f'_device|{key}|loss_per_sample'] = value
        if not key.startswith('indices') and key != 'overall':
            reduced[f'_device|{key}|_n'] = outputs[f'precondition|{key}'].sum()


def compute_device_metrics(reduced, any_tensor, metrics, outputs, targets):
    for metric_name, metric in metrics:
        path = f'{metric.prefix}{metric.task_name}'
        full_name = f'{path}|{metric_name}'
        metric_res = metric(outputs, targets)
        if isinstance(metric_res, dict):
            for key, value in metric_res.items():
                value = torch.as_tensor(value, dtype=any_tensor.dtype, device=any_tensor.device)
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                reduced[f'_device|{full_name}_{key}'] = value
        else:
            value = torch.as_tensor(metric_res, dtype=any_tensor.dtype, device=any_tensor.device)
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
            reduced[f'_device|{full_name}'] = value
        reduced[f'_device|{path}|_n'] = outputs[f'precondition|{metric.prefix}{metric.task_name}'].sum()


class GuidedGradCamPublisher:

    def __init__(self, model, layer, forward_pass_preprocess):
        from captum.attr import GuidedGradCam
        self.model = model
        self.grad_cam = GuidedGradCam(model=model, layer=layer)
        self.forward_pass_preprocess = forward_pass_preprocess

    def __call__(self, writer: SummaryWriter, tag, sample, idx):
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        if sample.shape[-1] == 3:  # H, W, C
            sample = sample.permute(2, 0, 1)
        X = self.forward_pass_preprocess(sample).unsqueeze(0)
        device = next(self.model.parameters()).device
        X = X.to(device)
        logits = self.model(X)
        res = self.grad_cam.attribute(X, logits.argmax())
        res = res.squeeze().detach().cpu().numpy()
        res = (res - res.min())
        res /= res.max()
        res = (res * 255).astype(np.uint8)
        writer.add_image(f"{tag}_gradcam", res, global_step=idx)


class SingleLossInterpretationCallback(IMetricCallback):
    def __init__(
            self,
            criterion,
            loaders_to_skip: Container[str] = (),
            prefix: str = "",
            input_key: str = "targets",
            output_key: str = "logits",
            idx_key=None,
            top_k=10,
            tensorboard_sequence: Sequence = None,
            tensorboard_publishers: Sequence[ITensorboardPublisher] = (),
            **loss_kwargs,
    ):
        super().__init__(prefix, input_key, output_key, **loss_kwargs)
        self.metric = criterion
        self.interpretations = {}
        self.top_k = top_k
        self.loggers = {}
        self.tensorboard_sequence = tensorboard_sequence
        self.tensorboard_publishers = tensorboard_publishers
        self._loaders_to_skip = loaders_to_skip
        self._idx_key = idx_key

    def _should_interpret_loader(self, runner: IRunner):
        if runner.loader_name in self._loaders_to_skip:
            return False
        if isinstance(runner.loaders[runner.loader_name].sampler, SequentialSampler):
            return True

        """
        If the sampler is not sequential, we cannot recover the original index of the sample,
        unless the user has provided `idx_key`.
        See: https://github.com/catalyst-team/catalyst/issues/950#issuecomment-703220633
        """
        return self._idx_key is not None

    def on_loader_start(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return
        if runner.loader_name not in self.loggers:
            logdir = runner.logdir / f"{runner.loader_name}_log"
            self.loggers[runner.loader_name] = SummaryWriter(str(logdir))
        if runner.loader_name not in self.interpretations:
            self.interpretations[runner.loader_name] = {
                "loss": [],
                "indices": [],
            }

    def on_loader_end(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return

        self.interpretations[runner.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.interpretations[runner.loader_name].items()
        }

        out_file = runner.logdir / f"{runner.loader_name}_interpretations.pkl"
        torch.save(self.interpretations[runner.loader_name], out_file)

        loss_sorter = self.interpretations[runner.loader_name]["loss"].argsort()
        indices_sorted = self.interpretations[runner.loader_name]["indices"][loss_sorter]
        indices = {
            "best": indices_sorted[: self.top_k],
            "worst": indices_sorted[-self.top_k:][::-1],
        }

        writer: SummaryWriter = self.loggers[runner.loader_name]
        for type_prefix in ["best", "worst"]:
            for idx in indices[type_prefix]:
                tag = f"{self.prefix}{type_prefix}"
                for tensorboard_publisher in self.tensorboard_publishers:
                    sample = self.tensorboard_sequence[idx]
                    tensorboard_publisher(writer, tag, sample, idx)

    def on_batch_end(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return
        if self.metric is None:
            return
        loss_items: torch.Tensor = self._compute_metric_value(runner.output, runner.input)
        if len(loss_items.shape) > 1:
            dims = tuple(range(1, len(loss_items.shape)))
            loss_items = loss_items.mean(dim=dims)

        if self._idx_key is None:
            bs = len(loss_items)
            indices_so_far = self.interpretations[runner.loader_name]["indices"]
            start_idx = (0 if len(indices_so_far) == 0 else (indices_so_far[-1][-1] + 1))
            indices = np.arange(start_idx, start_idx + bs)
        else:
            indices = runner.input[self._idx_key].detach().cpu().numpy()

        self.interpretations[runner.loader_name]["loss"].append(loss_items.detach().cpu().numpy())
        self.interpretations[runner.loader_name]["indices"].append(indices)

    @property
    def metric_fn(self):
        return self.metric
