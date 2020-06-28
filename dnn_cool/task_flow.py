class TaskInput:
    pass


class TaskOutput:

    def __init__(self):
        self.res = {}

    def __iadd__(self, other):
        pass


class Task:
    pass


class TaskFlow:

    def __init__(self, tasks):
        self.tasks = tasks

    def flow(self, x: TaskInput) -> TaskOutput:
        raise NotImplementedError()
