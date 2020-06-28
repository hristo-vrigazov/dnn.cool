import pytest

from dnn_cool.task_flow import TaskFlow, NestedResult, BinaryClassificationTask, ClassificationTask, \
    BinaryHardcodedTask, LocalizationTask, NestedClassificationTask, RegressionTask

import torch

from torch import nn


@pytest.fixture(scope='module')
def carsbg():
    class IsCarModule(nn.Module):

        def __init__(self, fc):
            super().__init__()
            self.fc = fc

        def forward(self, features, sliced):
            res = self.fc(features)
            res[sliced.decoded] = 1e3
            return res

    class IsCarTask(BinaryClassificationTask):

        def torch(self) -> nn.Module:
            return IsCarModule(super().torch())

    class CarsBgNetFlow(TaskFlow):

        def __init__(self):
            super().__init__(
                name='carsbg_flow',
                tasks=[
                    BinaryHardcodedTask(name='sliced'),
                    LocalizationTask(name='car_localization', module_options={
                        'in_features': 2560,
                        'bias': True
                    }),
                    LocalizationTask(name='license_plate_localization', module_options={
                        'in_features': 2560,
                        'bias': True
                    }),
                    IsCarTask(name='is_car', module_options={
                        'in_features': 2560,
                        'bias': True
                    }),
                    BinaryClassificationTask(name='has_license_plate', module_options={
                        'in_features': 2560,
                        'bias': True
                    }),
                    ClassificationTask(name='brand', module_options={
                        'in_features': 2560,
                        'out_features': 4,
                        'bias': True
                    }),
                    ClassificationTask(name='color', module_options={
                        'in_features': 2560,
                        'out_features': 26,
                        'bias': True
                    }),
                    RegressionTask(name='year', module_options={
                        'in_features': 2560,
                        'out_features': 1,
                        'bias': True
                    }),
                    NestedClassificationTask(name='model', top_k=5, module_options={
                        'in_features': 2560,
                        'out_features_nested': [9, 15, 2, 12],
                        'bias': True
                    })
                ])

        def flow(self, x, out):
            out += self.sliced(x.sliced)
            out += self.car_localization(x.car) | (~out.sliced)

            out += self.is_car(x.car, out.sliced)

            out += self.brand(x.common) | out.is_car
            out += self.color(x.common) | out.is_car
            out += self.year(x.common) | out.is_car

            out += self.model(x.common, out.brand) | out.is_car

            out += self.has_license_plate(x.lp) | out.is_car
            out += self.license_plate_localization(x.lp) | (out.has_license_plate & out.is_car)

            return out

    carsbg = CarsBgNetFlow()
    return carsbg


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
            'brand': torch.zeros(4, 1).long()
        }
    }

    module = module.train()
    res = module(example_dict)
    print(module)
    print(res)