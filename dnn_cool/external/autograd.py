

class IAutoGrad:

    def to_numpy(self, x):
        raise NotImplementedError()

    def as_float(self, x):
        raise NotImplementedError()
