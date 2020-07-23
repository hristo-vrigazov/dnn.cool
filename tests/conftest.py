import pytest
import torch
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, TensorDataset

from dnn_cool.modules import Identity
from dnn_cool.project import Project
from dnn_cool.task_flow import BinaryClassificationTask, TaskFlow, BinaryHardcodedTask, BoundedRegressionTask, \
    ClassificationTask, NestedClassificationTask, RegressionTask


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

    class Inputs(Dataset):

        def __init__(self, key, inputs):
            self.key = key
            self.inputs = inputs

        def __getitem__(self, item):
            return {
                self.key: self.inputs[item]
            }

        def __len__(self):
            return len(self.inputs)

    camera_blocked_task = BinaryClassificationTask('camera_blocked',
                                                   module=nn.Linear(n_features, out_features=1),
                                                   labels=camera_blocked.float().unsqueeze(dim=1))
    driver_seat_empty_task = BinaryClassificationTask('driver_seat_empty',
                                                      module=nn.Linear(n_features, 1),
                                                      labels=driver_seat_empty.float().unsqueeze(dim=1))
    passenger_seat_empty_task = BinaryClassificationTask('passenger_seat_empty',
                                                         module=nn.Linear(n_features, 1),
                                                         labels=passenger_seat_empty.float().unsqueeze(dim=1))
    driver_has_seatbelt_task = BinaryClassificationTask('driver_has_seatbelt',
                                                        module=nn.Linear(n_features, 1),
                                                        labels=driver_has_seatbelt.float().unsqueeze(dim=1))
    driver_uniform_type_task = ClassificationTask('driver_uniform_type',
                                                  module=nn.Linear(n_features, 3),
                                                  inputs=TensorDataset(inputs),
                                                  labels=driver_uniform_type.long())
    passenger_uniform_type_task = ClassificationTask('passenger_uniform_type',
                                                     module=nn.Linear(n_features, 3),
                                                     inputs=TensorDataset(inputs),
                                                     labels=passenger_uniform_type.long())

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

        def get_inputs(self, *args, **kwargs):
            return Inputs('inputs', inputs)

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


@pytest.fixture(scope='package')
def simple_df_project():
    df_data = [
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0 },
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1 },
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2 },
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3 },
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4 },
    ]

    df = pd.DataFrame(df_data)

    return Project(df, input_col='input', output_col=['camera_blocked', 'door_open', 'uniform_type'])
