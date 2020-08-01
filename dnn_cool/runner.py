from catalyst.dl import SupervisedRunner

from dnn_cool.task_flow import TaskFlow
from torch import optim

from functools import partial


class DnnCoolSupervisedRunner(SupervisedRunner):

    def __init__(self, task_flow: TaskFlow):
        super().__init__()
        self.task_flow = task_flow

        self.default_criterion = self.task_flow.get_loss()
        self.default_callbacks = self.default_criterion.catalyst_callbacks()
        self.default_optimizer = partial(optim.AdamW, lr=1e-4)

    def train(self, *args, **kwargs):
        kwargs['criterion'] = kwargs.get('criterion', self.default_criterion)
        callbacks = kwargs.get('callbacks', [])
        callbacks.extend(self.default_callbacks)
        kwargs['callbacks'] = callbacks

        if not 'optimizer' in kwargs:
            model = kwargs['model']
            optimizable_params = filter(lambda p: p.requires_grad, model.parameters())
            kwargs['optimizer'] = self.default_optimizer(params=optimizable_params)
        super().train(*args, **kwargs)
