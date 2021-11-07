from dataclasses import dataclass

from dnn_cool.converters.base import StatefulConverter
from dnn_cool.utils.base import Values


@dataclass()
class ValuesConverter(StatefulConverter):

    def to_values(self, df, col, guessed_type, perform_conversion=True):
        if col in self.col_mapping:
            converter = self.col_mapping[col]
            values = converter(df[col]) if perform_conversion else None
            return Values([col], [values], [guessed_type])
        if guessed_type in self.type_mapping:
            converter = self.type_mapping[guessed_type]
            values = converter(df[col]) if perform_conversion else None
            return Values([col], [values], [guessed_type])

        raise KeyError(f'Cannot convert column "{col}" from dataframe, because there is no registered converter'
                       f' for its guessed type "{guessed_type}".')