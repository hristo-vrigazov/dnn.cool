from typing import Union, Iterable, Optional, Dict, Callable


class Project:

    def __init__(self, df,
                 input_col: Union[str, Iterable[str]],
                 output_col: Union[str, Iterable[str]],
                 additional_converters: Optional[Dict[str, Callable]] = None):
        self._assert_col_in_df(input_col, df)
        self._assert_col_in_df(output_col, df)

        self.df = df
        self.input_col = input_col
        self.output_col = output_col
        if additional_converters is None:
            additional_converters = {}

    def _assert_col_in_df(self, col, df):
        if isinstance(col, str):
            assert col in self.df, self._not_found_error_message(col, df)
        else:
            for col_s in col:
                assert col_s in self.df, self._not_found_error_message(col_s, df)

    def _not_found_error_message(self, col, df):
        return f'Input column "{col}" not found in df. Dataframe has columns:' \
               f'{df.columns.tolist()}'
