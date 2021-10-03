import dnn_cool
from dnn_cool.tasks import BinaryClassificationTask
from dnn_cool.help import helper


def test_help_tasks():
    dnn_cool.help.show_help()
    task = BinaryClassificationTask('task', None)
