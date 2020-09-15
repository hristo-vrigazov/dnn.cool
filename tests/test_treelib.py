import torch


def test_treelib_regression(treelib_explanation_on_first_batch):
    actual = str(treelib_explanation_on_first_batch)
    expected = str(torch.load('./test_data/prediction_tree.pkl'))
    assert actual == expected
