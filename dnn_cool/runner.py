from catalyst.dl import SupervisedRunner, EarlyStoppingCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dnn_cool.task_flow import TaskFlow
from torch import optim

from functools import partial


class DnnCoolSupervisedRunner(SupervisedRunner):

    def __init__(self, task_flow: TaskFlow, early_stop: bool):
        super().__init__()
        self.task_flow = task_flow

        self.default_criterion = self.task_flow.get_loss()
        self.default_callbacks = self.default_criterion.catalyst_callbacks()
        self.default_optimizer = partial(optim.AdamW, lr=1e-4)
        self.default_scheduler = ReduceLROnPlateau

        if early_stop:
            self.default_callbacks.append(EarlyStoppingCallback(patience=5))

    def train(self, *args, **kwargs):
        kwargs['criterion'] = kwargs.get('criterion', self.default_criterion)
        kwargs['callbacks'] = kwargs.get('callbacks', self.default_callbacks)

        if not 'optimizer' in kwargs:
            model = kwargs['model']
            optimizable_params = filter(lambda p: p.requires_grad, model.parameters())
            kwargs['optimizer'] = self.default_optimizer(params=optimizable_params)

        if not 'scheduler' in kwargs:
            kwargs['scheduler'] = self.default_scheduler(kwargs['optimizer'])

        super().train(*args, **kwargs)
