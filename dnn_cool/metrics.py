from catalyst.utils.metrics import accuracy


def single_result_accuracy(*args, **kwargs):
    return accuracy(*args, **kwargs)[0]
