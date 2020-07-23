from typing import Union, Iterable, Optional, Dict, Callable


class MultiTaskProject:

    def __init__(self, df,
                 input_col: Union[str, Iterable[str]],
                 output_col: Union[str, Iterable[str]],
                 additional_converters: Optional[Dict[str, Callable]]=None):
        self.df = df
        self.input_col = input_col
        assert self.input_col in self.df
        self.output_col = output_col
        if additional_converters is None:
            additional_converters = []

