from typing import Optional, List, Callable

import torch
from torch import nn
from treelib import Tree

from dnn_cool.decoders.multilabel_classification import MultilabelClassificationDecoder
from dnn_cool.tasks.base import Task


class MultilabelClassificationTask(Task):

    def __init__(self, name, torch_module,
                 class_names: Optional[List[str]] = None,
                 dropout_mc=None):
        super().__init__(name,
                         torch_module,
                         activation=nn.Sigmoid(),
                         decoder=MultilabelClassificationDecoder(),
                         dropout_mc=dropout_mc)
        self.class_names = class_names

    def get_treelib_explainer(self) -> Callable:
        return self.generate_tree

    def generate_tree(self, task_name: str,
                      decoded: torch.Tensor,
                      activated: torch.Tensor,
                      logits: torch.Tensor,
                      node_identifier: str):
        tree = Tree()
        start_node = tree.create_node(task_name, node_identifier)
        for i, val in enumerate(decoded):
            name = i if self.class_names is None else self.class_names[i]
            description = f'{i}: {name} | decoded: {decoded[i]}, ' \
                          f'activated: {activated[i]:.4f}, ' \
                          f'logits: {logits[i]:.4f}'
            tree.create_node(description, f'{node_identifier}.{i}', parent=start_node)
        return tree, start_node
