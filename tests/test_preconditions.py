import torch

from dnn_cool.modules.torch import OnesCondition, LeafModuleOutput


def test_precondition_and_missing():
    is_category = OnesCondition('is_category')
    """
    path: str
    logits: torch.Tensor
    precondition: Condition
    activated: Optional[torch.Tensor]
    decoded: Optional[torch.Tensor]
    dropout_samples: Optional[torch.Tensor]
    """
    availability = OnesCondition('subcategory')
    leaf_output = LeafModuleOutput(
        path='subcategory',
        logits=torch.randn(4),
        precondition=availability,
        activated=torch.randn(4),
        decoded=torch.randn(4),
        dropout_samples=None
    )

    res = leaf_output | is_category
    actual = res.precondition.to_mask({
        '_availability': {
            'is_category': torch.tensor([True, True, False, True]),
            'subcategory': torch.tensor([False, True, False, True])
    }})

    expected = torch.tensor([False, True, False, True])

    assert all(expected == actual)
