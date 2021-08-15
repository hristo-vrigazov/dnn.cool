import time
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional

import joblib


class Verbosity(IntEnum):
    SILENT = 0
    BASIC_STATS = 1
    ALL = 2


def log(messsage):
    print(f'[dnn_cool] {messsage}.')


class StatsRegistry:

    def __init__(self, stats_file: Optional[Path] = None, verbosity = Verbosity.SILENT):
        self.stats_file = stats_file
        self.dct = self.get_dct()
        self.verbosity = verbosity

    def get_dct(self):
        if self.stats_file is None:
            return {}
        if not self.stats_file.exists():
            return {}
        return joblib.load(self.stats_file)

    def dump(self):
        joblib.dump(self.dct, self.stats_file)


class Logger:

    def __init__(self, key, stats_registry: StatsRegistry):
        self.key = key
        self.stats_registry = stats_registry
        self.start_t = None
        self.end_t = None

    def __enter__(self):
        self.start_t = time.time()
        if self.stats_registry.verbosity >= Verbosity.BASIC_STATS:
            log(f'Starting "{self.key}", last time execution took {self.stats_registry.dct.get(self.key)}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_t = time.time()
        if self.stats_registry.verbosity >= Verbosity.BASIC_STATS:
            self.stats_registry.dct[self.key] = self.end_t - self.start_t
            log(f'Finished "{self.key}, execution took {self.stats_registry.dct.get(self.key)}')
            self.stats_registry.dump()

