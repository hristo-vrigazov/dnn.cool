
class IFeaturesDict:

    def __getattr__(self, item):
        raise NotImplementedError()


class ICondition:
    def __invert__(self):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()


class IOut:

    def __iadd__(self, other):
        raise NotImplementedError()

    def __getattr__(self, item) -> ICondition:
        raise NotImplementedError()


class IFlowTaskResult:

    def __or__(self, precondition: ICondition):
        raise NotImplementedError()


class IFlowTask:

    def __call__(self, *args, **kwargs) -> IFlowTaskResult:
        raise NotImplementedError()

