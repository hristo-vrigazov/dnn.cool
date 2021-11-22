import functools

from dnn_cool.verbosity import log


class Helper:

    def __init__(self):
        self.show_help = False
        self.already_shown = set()

    def __call__(self, original_function=None, before_type=None, after_type=None):
        def _decorate(function):
            @functools.wraps(function)
            def wrapped_function(*args, **kwargs):
                if self.show_help and before_type is not None:
                    self.before_callback(before_type, function, args, kwargs)
                r = function(*args, **kwargs)
                if self.show_help and after_type is not None:
                    self.after_callback(after_type, function, args, kwargs, r)
                return r
            return wrapped_function

        if original_function:
            return _decorate(original_function)

        return _decorate

    def before_callback(self, before_type, function, args, kwargs):
        pass

    def after_callback(self, after_type, function, args, kwargs, r):
        if after_type in self.already_shown:
            return
        if after_type == 'task':
            log('You created a task! You can create more tasks in a similar way.')
            log('Once done, you can create a TaskFlow - a special kind of task, that consists of smaller tasks.')
            log('To do this, first create an instance of the Tasks class:')
            print('tasks = Tasks([<list of the leaf tasks created here ...>])')
            self.already_shown.add('task')
            return
        if after_type == 'tasks':
            log('You created a Tasks instance! You can now add task flows.')
            log('An example task flow:')
            flow_str = """
@tasks.add_flow
def localize_flow(flow, x, out):
    out += flow.obj_exists(x.features)
    out += flow.obj_x(x.features) | out.obj_exists
    out += flow.obj_y(x.features) | out.obj_exists
    out += flow.obj_w(x.features) | out.obj_exists
    out += flow.obj_h(x.features) | out.obj_exists
    out += flow.obj_class(x.features) | out.obj_exists
    return out
"""
            print(flow_str)
            log('To do this, you can use the "add_flow" method, which can be used as a decorator to a function.')
            log('The task flow function accepts 3 arguments: the flow, the input features, and the output.')
            log('In the task flow function, you can use the currently registered tasks as attributes.')
            log('The input features are a dict-like object, and you can access features as attributes.')
            log('The output is an object, which you must return in the end, and you can add into using "+=" operator.')
            log('If a task is preconditioned on another task, use the "|" operator.')
            self.already_shown.add('tasks')
            return
        if after_type == 'tasks.add_flow':
            log('You added a task flow! You can add more task flows in the same way.')
            log('A task flow is a task itself, so you can use your currently registered task flows in new task flows!')
            return


helper = Helper()


def show():
    helper.show_help = True
    log("Welcome to dnn_cool help!")
    log("The central abstraction in dnn_cool is a task.")
    log("The built-in tasks are located in the `dnn_cool.tasks` module.")
    log("For example, to import all built-in tasks, run:")
    print("from dnn_cool.tasks import *")
    log("After that, you can create a task by supplying a name and a PyTorch module, for example:")
    print("is_task = BinaryClassificationTask(<task name here ...>, <PyTorch module here ...>)")
    log("You can also create your own tasks, by creating an instance of the Task class.")


def hide():
    helper.show_help = False
