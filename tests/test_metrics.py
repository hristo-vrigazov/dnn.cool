from sklearn.metrics import accuracy_score

from dnn_cool.metrics import BinaryAccuracy, NumpyMetric


def test_binary_accuracy(simple_binary_data):
    x, y, task_mock = simple_binary_data
    metric = BinaryAccuracy()
    metric.bind_to_task(task_mock)

    res = metric(x, y, activate=True)[0].item()
    assert res >= 0.70


def test_scikit_accuracy(simple_binary_data):
    x, y, task_mock = simple_binary_data
    metric = NumpyMetric(accuracy_score)
    metric.bind_to_task(task_mock)

    res = metric(x, y, activate=True).item()
    assert res >= 0.70

