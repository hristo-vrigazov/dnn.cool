
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
    """
    This class represents the inferface that a class has to implement to be able to act asa a precondition to a
    :class:`IFlowTaskResult`.
    """
    def __invert__(self):
        """
        Condition inversion.

        :return: the inverted mask

        """
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()


class IOut:
    """
    Ths interface is used to accumulate the result in a TaskFlow.
    """

    def __iadd__(self, other):
        raise NotImplementedError()

    def __getattr__(self, item) -> ICondition:
        """
        Accesses the result for a given task name.

        :param item: the full name of the task

        :return:
        """
        raise NotImplementedError()


class IFlowTaskResult:
    """
    This interface represents the result of a task flow.
    """

    def __or__(self, precondition: ICondition):
        """
        Make the supplied :class:`ICondition` a precondition for the result of this task.

        :param precondition: the precondition.

        :return: self with precondition added.
        """
        raise NotImplementedError()


class IFlowTask:

    def __call__(self, *args, **kwargs) -> IFlowTaskResult:
        raise NotImplementedError()

