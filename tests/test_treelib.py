from pathlib import Path

import torch
import numpy as np

def test_treelib_regression(treelib_explanation_on_first_batch):
    actual = str(treelib_explanation_on_first_batch)
    expected_path = Path('./test_data/prediction_tree.pkl')
    if not expected_path.exists():
        expected_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(actual, expected_path)
    expected = str(torch.load(expected_path))
    np.set_printoptions(precision=4)
    print(actual)
    assert actual == expected
