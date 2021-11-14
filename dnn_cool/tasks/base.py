from typing import Optional, Callable

from torch import nn

from dnn_cool.decoders.base import Decoder
from dnn_cool.help import helper
from dnn_cool.treelib import default_leaf_tree_explainer


class IMinimal:

    def get_minimal(self):
        raise NotImplementedError()


class Task(IMinimal):

    @helper(after_type='task')
    def __init__(self, name, torch_module, activation, decoder, dropout_mc, treelib_explainer=None):
        self.name = name
        self.activation = activation
        self.decoder = decoder
        self.torch_module = torch_module
        self.dropout_mc = dropout_mc
        self.treelib_explainer = treelib_explainer

    def get_name(self) -> str:
        return self.name

    def get_activation(self) -> Optional[nn.Module]:
        return self.activation

    def get_decoder(self) -> Decoder:
        return self.decoder

    def has_children(self) -> bool:
        return False

    def is_train_only(self) -> bool:
        return False

    def torch(self):
        return self.torch_module

    def get_dropout_mc(self):
        return self.dropout_mc

    def get_treelib_explainer(self) -> Callable:
        return default_leaf_tree_explainer if self.treelib_explainer is None else self.treelib_explainer

    def get_minimal(self):
        return self
