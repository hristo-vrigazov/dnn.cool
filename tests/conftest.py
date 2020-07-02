from typing import Dict

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from dnn_cool.datasets import FlowDataset
from dnn_cool.modules import Identity
from dnn_cool.task_flow import BinaryClassificationTask, TaskFlow, BinaryHardcodedTask, LocalizationTask, \
    ClassificationTask, RegressionTask, NestedClassificationTask, RegressionTask


@pytest.fixture(scope='package')
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
            base_options = {
                'in_features': 2560,
                'bias': True
            }
            super().__init__(
                name='carsbg_flow',
                tasks=[
                    BinaryHardcodedTask(name='sliced'),
                    LocalizationTask(name='car_localization', module_options=base_options),
                    LocalizationTask(name='license_plate_localization', module_options=base_options),
                    IsCarTask(name='is_car', module_options=base_options),
                    BinaryClassificationTask(name='has_license_plate', module_options=base_options),
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
                    }, activation_func=nn.Sigmoid()),
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


@pytest.fixture(scope='package')
def yolo_anchor():
    class YoloAnchorFlow(TaskFlow):

        def __init__(self):
            super().__init__(
                name='yolo_flow',
                tasks=[
                    BinaryClassificationTask(name='has_object', module_options={
                        'in_features': 2560,
                        'bias': True
                    }),
                    RegressionTask(name='xy', module_options={
                        'in_features': 2560,
                        'out_features': 2,
                        'bias': True
                    }, activation_func=nn.Sigmoid()),
                    RegressionTask(name='wh', module_options={
                        'in_features': 2560,
                        'out_features': 2,
                        'bias': True
                    }, activation_func=Identity()),
                    ClassificationTask(name='object_class', module_options={
                        'in_features': 2560,
                        'out_features': 10,
                        'bias': True
                    })
                ])

        def flow(self, x, out):
            out += self.has_object(x.features)
            out += self.xy(x.features) | out.has_object
            out += self.wh(x.features) | out.has_object
            out += self.object_class(x.features) | out.has_object
            return out

    carsbg = YoloAnchorFlow()
    return carsbg


@pytest.fixture(scope='package')
def nested_carsbg(carsbg):
    module_options = {
        'in_features': 2560,
        'bias': True
    }
    camera_blocked = BinaryClassificationTask(name='camera_blocked', module_options=module_options)

    class CameraBlockedCarsBgFlow(TaskFlow):

        def __init__(self):
            super().__init__('camera_blocked_carsbg_flow', [camera_blocked, carsbg])

        def flow(self, x, out):
            out += self.camera_blocked(x.common)
            out += self.carsbg_flow(x) | out.camera_blocked
            return out

    return CameraBlockedCarsBgFlow()


@pytest.fixture(scope='package')
def interior_car_task():
    n = 2 ** 10
    n_features = 128

    # Generate ground truth
    camera_blocked = torch.randn(n).float() > 0.
    passenger_seat_empty = torch.randn(n).float() > -0.3
    driver_seat_empty = torch.randn(n).float() > 0.3

    driver_has_seatbelt = torch.randn(n).float() > 0.
    # 0 - police, 1 - doctor, 2 - civilian
    driver_uniform_type = torch.randint(low=0, high=3, size=(n,)).long()
    passenger_uniform_type = torch.randint(low=0, high=3, size=(n,)).long()

    inputs = torch.stack((camera_blocked.float(),
                          passenger_seat_empty.float(),
                          driver_seat_empty.float(),
                          driver_has_seatbelt.float(),
                          driver_uniform_type.float() * 3.,
                          passenger_uniform_type.float() * 7.), dim=1)

    class BinaryThresholdedTask(BinaryClassificationTask):

        def __init__(self, name, labels):
            super().__init__(name, module_options={'in_features': 128})
            self.labels = labels

        def datasets(self, **kwargs) -> Dataset:
            return TensorDataset(inputs, self.labels)

    class DummyClassificationTask(ClassificationTask):

        def __init__(self, name: str, labels):
            super().__init__(name, module_options={
                'in_features': n_features,
                'out_features': 3,
                'bias': True
            })
            self.inputs = inputs
            self.labels = labels

        def datasets(self, **kwargs) -> Dataset:
            return TensorDataset(inputs, self.labels)

    camera_blocked_task = BinaryThresholdedTask('camera_blocked', camera_blocked.float())
    driver_seat_empty_task = BinaryThresholdedTask('driver_seat_empty', driver_seat_empty.float())
    passenger_seat_empty_task = BinaryThresholdedTask('passenger_seat_empty', passenger_seat_empty.float())
    driver_has_seatbelt_task = BinaryThresholdedTask('driver_has_seatbelt', driver_has_seatbelt.float())
    driver_uniform_type_task = DummyClassificationTask('driver_uniform_type', driver_uniform_type.long())
    passenger_uniform_type_task = DummyClassificationTask('passenger_uniform_type', passenger_uniform_type.long())

    class DriverFlow(TaskFlow):

        def __init__(self):
            super().__init__('driver_flow', [driver_seat_empty_task,
                                             driver_has_seatbelt_task,
                                             driver_uniform_type_task])

        def flow(self, x, out):
            out += self.driver_seat_empty(x.driver_features)
            out += self.driver_has_seatbelt(x.driver_features) | (~out.driver_seat_empty)
            out += self.driver_uniform_type(x.driver_features) | (~out.driver_seat_empty)
            return out

    driver_flow = DriverFlow()

    class PassengerFlow(TaskFlow):

        def __init__(self):
            super().__init__('passenger_flow', [passenger_seat_empty_task, passenger_uniform_type_task])

        def flow(self, x, out):
            out += self.passenger_seat_empty(x.passenger_features)
            out += self.passenger_uniform_type(x.passenger_features) | out.passenger_seat_empty
            return out

    passenger_flow = PassengerFlow()

    class FullFlow(TaskFlow):

        def __init__(self):
            super().__init__('full_flow', [camera_blocked_task, driver_flow, passenger_flow])

        def flow(self, x, out):
            out += self.camera_blocked(x.features)
            out += self.driver_flow(x) | (~out.camera_blocked)
            out += self.passenger_flow(x) | (~out.camera_blocked)
            return out

    flow = FullFlow()

    class InteriorMonitorModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(6, 64, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=True),
                nn.ReLU(inplace=True),
            )
            self.driver_features_fc = nn.Linear(64, n_features, bias=True)
            self.passenger_features_fc = nn.Linear(64, n_features, bias=True)
            self.features_fc = nn.Linear(64, n_features, bias=True)

            self.flow_module = flow.torch()

        def forward(self, x):
            common = self.seq(x['inputs'])

            res = {
                'driver_features': self.driver_features_fc(common),
                'passenger_features': self.passenger_features_fc(common),
                'features': self.features_fc(common),
                'gt': x['gt']
            }
            return self.flow_module(res)

    return InteriorMonitorModule(), flow
