from dnn_cool.task_flow import NestedResult

import torch


def test_trace_flow(carsbg):
    task_input = NestedResult(carsbg)
    task_input.res['car'] = 2560
    task_input.res['common'] = 2560
    task_input.res['lp'] = 2560
    task_input.res['sliced'] = True

    r = carsbg.trace_flow(task_input)

    print('Final res')
    print(r)


def test_eval_flow(carsbg):
    module = carsbg.torch()

    example_dict = {
        'car': torch.ones(4, 2560).float(),
        'common': torch.ones(4, 2560).float(),
        'lp': torch.ones(4, 2560).float(),
        'sliced': torch.tensor([True, False, True, False]).bool()
    }

    module = module.eval()
    res = module(example_dict)
    print(module)
    print(res)


def test_train_flow(carsbg):
    module = carsbg.torch()

    example_dict = {
        'car': torch.ones(4, 2560).float(),
        'common': torch.ones(4, 2560).float(),
        'lp': torch.ones(4, 2560).float(),
        'sliced': torch.tensor([True, False, True, False]).bool(),
        'gt': {
            'is_car': torch.ones(4).bool(),
            'brand': torch.zeros(4, 1).long(),
            'has_lp': torch.tensor([True, True, True, False]).bool(),
        }
    }

    module = module.train()
    res = module(example_dict)
    print(module)
    print(res)


def test_train_flow_nested(nested_carsbg):
    module = nested_carsbg.torch()

    example_dict = {
        'car': torch.ones(4, 2560).float(),
        'common': torch.ones(4, 2560).float(),
        'lp': torch.ones(4, 2560).float(),
        'sliced': torch.tensor([True, False, True, False]).unsqueeze(dim=1).bool(),
        'gt': {
            'is_car': torch.ones(4, 1).bool(),
            'brand': torch.zeros(4, 1).long(),
            'has_lp': torch.tensor([True, True, True, False]).unsqueeze(dim=1).bool(),
        }
    }

    module = module.train()
    res = module(example_dict)
    print(module)
    print(res)