import dnn_cool
from dnn_cool.tasks import BinaryClassificationTask, Tasks
from dnn_cool.help import helper


def test_help_tasks():
    dnn_cool.help.show()
    task = BinaryClassificationTask('task', None)

    tasks = Tasks([task])

    @tasks.add_flow
    def task_flow(flow, x, out):
        out += flow.task(x.features)
        return out
