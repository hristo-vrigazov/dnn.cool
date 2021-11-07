from pathlib import Path
from typing import Union

import numpy as np

from dnn_cool.memmap.base import RaggedMemoryMap


class StringsMemmap(RaggedMemoryMap):

    def __init__(self, path: Union[str, Path], shapes, dtype=np.int64, mode='r+',
                 save_metadata=True, initialization_data=None):
        super().__init__(path, shapes, dtype, mode=mode,
                         save_metadata=save_metadata,
                         initialization_data=initialization_data)

    @classmethod
    def from_list_of_strings(cls, path, list_of_strings, dtype=np.int64):
        list_of_unicode_ints = []
        shapes = []
        for string in list_of_strings:
            unicode_list = [ord(c) for c in string]
            list_of_unicode_ints.append(unicode_list)
            shapes.append(len(unicode_list))
        return cls(path=path,
                   shapes=shapes,
                   dtype=dtype,
                   mode='w+',
                   initialization_data=list_of_strings)

    def set_single_index(self, key, value):
        super().set_single_index(key, np.array([ord(c) for c in value]))

    def get_single_index(self, item):
        unicode_result = super(StringsMemmap, self).get_single_index(item)
        return ''.join([chr(i) for i in unicode_result])