import torch
from transformers.modeling_bert import BertOnlyMLMHead

from dnn_cool.external.torch import TorchAutoGrad
from dnn_cool.losses.torch import ReducedPerSample, LanguageModelCrossEntropyLoss
from dnn_cool.missing_values import all_correct
from dnn_cool.tasks.base import Task
from dnn_cool.tasks.development.task_flow import TaskForDevelopment


class MaskedLanguageModelingTask(Task):

    def __init__(self, name, config, dropout_mc=None):
        torch_module = BertOnlyMLMHead(config)
        super().__init__(name, torch_module, activation=None, decoder=None, dropout_mc=dropout_mc)

    def is_train_only(self) -> bool:
        return True


class MaskedLanguageModelingTaskForDevelopment(TaskForDevelopment):

    def __init__(self, name: str, labels):
        per_sample = ReducedPerSample(LanguageModelCrossEntropyLoss(reduction='none'))
        super().__init__(name, labels,
                         criterion=LanguageModelCrossEntropyLoss(),
                         per_sample_criterion=per_sample,
                         available_func=all_correct,
                         metrics=[],
                         autograd=TorchAutoGrad())

