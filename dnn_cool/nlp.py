from dataclasses import dataclass

import torch
from transformers.modeling_bert import BertOnlyMLMHead

from dnn_cool.losses import LanguageModelCrossEntropyLoss, ReducedPerSample
from dnn_cool.missing_values import all_correct
from dnn_cool.tasks import Task


@dataclass()
class MaskedLanguageModelingTask(Task):

    def __init__(self, name: str, labels, config, inputs=None):
        kwargs = {
            'name': name,
            'labels': labels,
            'loss': LanguageModelCrossEntropyLoss(),
            'per_sample_loss': ReducedPerSample(LanguageModelCrossEntropyLoss(reduction='none'), reduction=torch.mean),
            'available_func': all_correct,
            'inputs': inputs,
            'activation': None,
            'decoder': None,
            'module': BertOnlyMLMHead(config),
            'metrics': ()
        }
        super().__init__(**kwargs)

    def is_train_only(self) -> bool:
        return True
