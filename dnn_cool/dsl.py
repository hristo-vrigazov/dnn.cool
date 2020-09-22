
class IFeaturesDict:
    """ This class represents the interface that a class has to implement to be an intermediary for the results
    in :class:`dnn_cool.task_flow.TaskFlow`.

    """

    def __getattr__(self, item):
        """ This will have different meaning depending on the implementation. For example, inside a `nn.Module`, this
        class would be wrapping a dictionary and this method select this as a key in the wrapped dictionary.
        :param item: attribute name
        """
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

