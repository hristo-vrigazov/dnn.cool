import functools


class Helper:

    def __init__(self):
        self.show_help = False

    def __call__(self, original_function=None, before_message=None, after_message=None):
        def _decorate(function):
            @functools.wraps(function)
            def wrapped_function(*args, **kwargs):
                if self.show_help and before_message is not None:
                    print(before_message)
                r = function(*args, **kwargs)
                if self.show_help and after_message is not None:
                    print(after_message)
                return r
            return wrapped_function

        if original_function:
            return _decorate(original_function)

        return _decorate


helper = Helper()


def show_help():
    helper.show_help = True
