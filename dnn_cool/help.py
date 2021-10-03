class Helper:

    def __init__(self):
        self.show_help = False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.show_help:
                print("Something is happening before the function is called.")
            r = func(*args, **kwargs)
            if self.show_help:
                print("Something is happening after the function is called.")
            return r

        return wrapper


helper = Helper()


def show_help():
    helper.show_help = True
