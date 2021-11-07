from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import joblib

from dnn_cool.verbosity import StatsRegistry, Logger


def extract_state_when_possible(mapping):
    out_mapping = {}
    for key, value in mapping.items():
        try:
            out_mapping[key] = value.state_dict()
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return out_mapping


def load_state_from_directory_when_possible(mapping: Dict, directory: Path, stats_registry):
    for key, value in mapping.items():
        try:
            if (directory / f'{key}.pkl').exists():
                with Logger(f'Load "{directory}/{key}.pkl"', stats_registry=stats_registry):
                    if hasattr(value, 'load_from_system'):
                        mapping[key].load_from_system(directory, key)
                    else:
                        state_dict = joblib.load(directory / f'{key}.pkl')
                        mapping[key].load_state_dict(state_dict)
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass
    return mapping


def dump_state_to_directory_when_possible(mapping, directory, stats_registry):
    for key, value in mapping.items():
        try:
            directory.mkdir(exist_ok=True)
            with Logger(f'Dump "{directory}/{key}.pkl"', stats_registry=stats_registry):
                if hasattr(value, 'dump_to_filesystem'):
                    value.dump_to_filesystem(directory, key)
                else:
                    joblib.dump(value.state_dict(), directory / f'{key}.pkl')
        except AttributeError:
            # It is ok for a converter not to have a state at all.
            pass


def not_found_error_message(col, df):
    return f'Input column "{col}" not found in df. Dataframe has columns: {df.columns.tolist()}'


def assert_col_in_df(df, col):
    if isinstance(col, str):
        assert col in df, not_found_error_message(col, df)
    else:
        for col_s in col:
            assert col_s in df, not_found_error_message(col_s, df)


@dataclass()
class TypeGuesser:
    type_mapping: Dict = field(default_factory=lambda: {})

    def guess(self, df, output_col):
        if output_col in self.type_mapping:
            return self.type_mapping[output_col]
        if str(df[output_col].dtype) == 'bool':
            return 'binary'

        if str(df[output_col].dtype).startswith('int'):
            return 'category'

    def state_dict(self):
        return self.type_mapping

    def load_state_from_directory(self, directory: Path):
        mapping_file = directory / 'type_mapping.pkl'
        if not mapping_file.exists():
            return
        self.type_mapping = joblib.load(mapping_file)

    def dump_state_to_directory(self, directory):
        directory.mkdir(exist_ok=True)
        joblib.dump(self.type_mapping, directory / 'type_mapping.pkl')


@dataclass()
class StatefulConverter:
    col_mapping: Dict = field(default_factory=lambda: {})
    type_mapping: Dict = field(default_factory=lambda: {})
    stats_registry: StatsRegistry = field(default_factory=StatsRegistry)

    def state_dict(self):
        return {
            'col': extract_state_when_possible(self.col_mapping),
            'type': extract_state_when_possible(self.type_mapping)
        }

    def dump_state_to_directory(self, directory: Path):
        directory.mkdir(exist_ok=True)
        dump_state_to_directory_when_possible(self.col_mapping, directory / 'col', self.stats_registry)
        dump_state_to_directory_when_possible(self.type_mapping, directory / 'type', self.stats_registry)

    def load_state_from_directory(self, directory: Path):
        load_state_from_directory_when_possible(self.col_mapping, directory / 'col', self.stats_registry)
        load_state_from_directory_when_possible(self.type_mapping, directory / 'type', self.stats_registry)

    def connect_to_stats_registry(self, stats_registry):
        self.stats_registry = stats_registry
