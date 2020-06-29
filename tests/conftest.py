import pytest
from torch import nn

from dnn_cool.modules import Identity
from dnn_cool.task_flow import BinaryClassificationTask, TaskFlow, BinaryHardcodedTask, LocalizationTask, \
    ClassificationTask, RegressionTask, NestedClassificationTask


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
