from typing import Iterable

from torch import nn

from dnn_cool.modules import SigmoidEval, SoftmaxEval


class Result:

    def __init__(self):
        self.precondition = None

    def __or__(self, result):
        self.precondition = result
        return self


class BooleanResult(Result):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __invert__(self):
        print('Invert!')
        return self

    def __and__(self, other):
        print('And')
        return self


class LocalizationResult(Result):
    def __init__(self, *args, **kwargs):
        super().__init__()


class ClassificationResult(Result):
    def __init__(self, *args, **kwargs):
        super().__init__()


class NestedClassificationResult(Result):
    def __init__(self, *args, **kwargs):
        super().__init__()


class RegressionResult(Result):
    def __init__(self, *args, **kwargs):
        super().__init__()


class NestedResult(Result):

    def __init__(self, key=None, value=None):
        super().__init__()
        self.res = {}
        if key is not None:
            self.res[key] = value

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


class Task:

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs) -> Result:
        return NestedResult(key=self.name, value=self.do_call(*args, **kwargs))

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
        return BooleanResult(self, args, kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class LocalizationTask(Task):
    def do_call(self, *args, **kwargs):
        return LocalizationResult(self, args, kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class BinaryClassificationTask(Task):

    def do_call(self, *args, **kwargs) -> BooleanResult:
        return BooleanResult(self, args, kwargs)

    def activation(self) -> nn.Module:
        return SigmoidEval()

    def loss(self):
        return nn.BCEWithLogitsLoss()


class ClassificationTask(Task):

    def do_call(self, *args, **kwargs):
        return ClassificationResult(self, args, kwargs)

    def activation(self) -> nn.Module:
        return SoftmaxEval()

    def loss(self):
        return nn.CrossEntropyLoss()


class RegressionTask(Task):

    def do_call(self, *args, **kwargs):
        return RegressionResult(self, args, kwargs)

    def activation(self) -> nn.Module:
        pass

    def loss(self):
        pass


class NestedClassificationTask(Task):

    def __init__(self, name, top_k):
        super().__init__(name)
        self.top_k = top_k

    def do_call(self, *args, **kwargs):
        return NestedClassificationResult(self, args, kwargs)

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

