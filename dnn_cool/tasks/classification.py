from typing import Optional, List, Callable

import torch
from torch import nn
from treelib import Tree

from dnn_cool.decoders.classification import ClassificationDecoder
from dnn_cool.tasks.base import Task


class ClassificationTask(Task):

    def __init__(self, name, torch_module,
                 class_names: Optional[List[str]] = None,
                 top_k: Optional[int] = 5,
                 dropout_mc=None):
        super().__init__(name,
                         torch_module=torch_module,
                         activation=nn.Softmax(dim=-1),
                         decoder=ClassificationDecoder(),
                         dropout_mc=dropout_mc)
        self.class_names: List[str] = class_names
        self.top_k = top_k

    def get_treelib_explainer(self) -> Callable:
        return self.generate_tree

    def generate_tree(self, task_name: str,
                      decoded: torch.Tensor,
                      activated: torch.Tensor,
                      logits: torch.Tensor,
                      node_identifier: str):
        tree = Tree()
        start_node = tree.create_node(task_name, node_identifier)
        for i, idx in enumerate(decoded[:self.top_k]):
            name = idx if self.class_names is None else self.class_names[idx]
            description = f'{i}: {name} | activated: {activated[idx]:.4f}, logits: {logits[idx]:.4f}'
            tree.create_node(description, f'{node_identifier}.{idx}', parent=start_node)
        return tree, start_node
