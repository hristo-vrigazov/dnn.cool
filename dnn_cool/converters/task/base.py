from dataclasses import dataclass

from dnn_cool.converters.base import StatefulConverter


@dataclass()
class TaskConverter(StatefulConverter):

    def to_task(self, output_col, guessed_type, values):
        if output_col in self.col_mapping:
            return self.col_mapping[output_col](name=output_col, labels=values)

        return self.type_mapping[guessed_type](name=output_col, labels=values)