import os

from typing import Iterable

from torch import nn

from dnn_cool.modules import SigmoidEval, SoftmaxEval


class Result:

    def __init__(self, task, *args, **kwargs):
        self.precondition = None
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def __or__(self, result):
        self.precondition = result
        return self

    def __repr__(self):
        args = map(str, self.args)
        kwargs = map(str, self.kwargs)
        arguments = ', '.join(args) + ', '.join(kwargs)
        precondition = f' | {self.precondition}' if self.precondition is not None else ''
        return f'{self.task.get_name()}({arguments}){precondition}'


class BooleanResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)

    def __invert__(self):
        return self

    def __and__(self, other):
        return self


class LocalizationResult(Result):
    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class ClassificationResult(Result):
    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class NestedClassificationResult(Result):
    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class RegressionResult(Result):
    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)


class NestedResult(Result):

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        self.res = {}
        key = kwargs.get('key', None)
        if key is not None:
            self.res[key] = kwargs.get('value', None)

    def __iadd__(self, other):
        self.res.update(other.res)
        return self

    def __getattr__(self, attr):
        return self.res[attr]

    def __or__(self, result):
        super().__or__(result)
        for key, value in self.res.items():
            self.res[key] = value | result
        return self

    def __repr__(self):
        repr = f'{{{os.linesep}'
        for key, value in self.res.items():
            repr += f'\t{key}: {value}{os.linesep}'
        repr += f'{os.linesep}}}'
        return repr


class Task:

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs) -> Result:
        return NestedResult(task=self, key=self.name, value=self.do_call(*args, **kwargs))

    def do_call(self, *args, **kwargs):
        raise NotImplementedError()

    def activation(self) -> nn.Module:
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()

    def get_name(self):
        return self.name


class BinaryHardcodedTask(Task):
    def do_call(self, *args, **kwargs) -> BooleanResult:
        return BooleanResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class LocalizationTask(Task):
    def do_call(self, *args, **kwargs):
        return LocalizationResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class BinaryClassificationTask(Task):

    def do_call(self, *args, **kwargs) -> BooleanResult:
        return BooleanResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        return SigmoidEval()

    def loss(self):
        return nn.BCEWithLogitsLoss()


class ClassificationTask(Task):

    def do_call(self, *args, **kwargs):
        return ClassificationResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        return SoftmaxEval()

    def loss(self):
        return nn.CrossEntropyLoss()


class RegressionTask(Task):

    def do_call(self, *args, **kwargs):
        return RegressionResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class NestedClassificationTask(Task):

    def __init__(self, name, top_k):
        super().__init__(name)
        self.top_k = top_k

    def do_call(self, *args, **kwargs):
        return NestedClassificationResult(self, *args, **kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class TaskFlow(Task):

    def __init__(self, name, tasks: Iterable[Task]):
        super().__init__(name)
        self.tasks = {}
        for task in tasks:
            self.tasks[task.get_name()] = task

    def flow(self, x: NestedResult) -> NestedResult:
        raise NotImplementedError()

    def __call__(self, x: NestedResult) -> NestedResult:
        return self.flow(x)

    def __getattr__(self, attr):
        return self.tasks[attr]

