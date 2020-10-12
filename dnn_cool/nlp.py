from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_bert import BertOnlyMLMHead

from dnn_cool.decoders import ClassificationDecoder
from dnn_cool.losses import LanguageModelCrossEntropyLoss, ReducedPerSample
from dnn_cool.missing_values import positive_values
from dnn_cool.task_flow import Task


@dataclass()
class MaskedLanguageModelingTask(Task):

    def __init__(self, name: str, labels, config, inputs=None):
        kwargs = {
            'name': name,
            'labels': labels,
            'loss': LanguageModelCrossEntropyLoss(),
            'per_sample_loss': ReducedPerSample(LanguageModelCrossEntropyLoss(reduction='none'), reduction=torch.mean),
            'available_func': positive_values,
            'inputs': inputs,
            'activation': nn.Softmax(dim=-1),
            'decoder': ClassificationDecoder(),
            'module': BertOnlyMLMHead(config),
            'metrics': ()
        }
        super().__init__(**kwargs)
