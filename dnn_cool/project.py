

class MultiTaskProject:

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_df(cls, df):
        return cls(df)
