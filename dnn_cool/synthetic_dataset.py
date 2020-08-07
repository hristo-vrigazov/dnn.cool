from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2
import torch

from functools import partial

from torch.utils.data import DataLoader

from dnn_cool.converters import TypeGuesser, ValuesConverter, TaskConverter, Converters
from dnn_cool.decoders import Decoder
from dnn_cool.project import Project
from dnn_cool.task_flow import BoundedRegressionTask, BinaryClassificationTask, TaskFlow
from dnn_cool.utils import torch_split_dataset
from dnn_cool.value_converters import binary_value_converter
from torch import nn


def generate_camera_blocked_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    # randomly add some noise
    n = 40
    i = np.random.randint(0, 64, size=n)
    j = np.random.randint(0, 64, size=n)
    c = np.random.randint(0, 3, size=n)
    values = np.random.randint(0, 255, size=n)
    img[i, j, c] = values

    return img, {'camera_blocked': True}


def generate_door_open_image():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255

    # randomly add some noise
    n = 40
    i = np.random.randint(0, 64, size=n)
    j = np.random.randint(0, 64, size=n)
    c = np.random.randint(0, 3, size=n)
    values = np.random.randint(0, 255, size=n)
    img[i, j, c] = values

    return img, {'camera_blocked': False, 'door_open': True, 'person_present': False}


def generate_door_closed_image(door_locked):
    img, res = generate_door_open_image()

    offsets = np.random.randint(-10, 10, size=4)
    x1, y1 = int(10 + offsets[0]), int(10 + offsets[1])
    x2, y2 = int(50 + offsets[2]), int(50 + offsets[2])

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(139, 69, 19), thickness=-1)

    if door_locked:
        lock_start = x1, int((y1 + y2) / 2)
        lock_end = lock_start[0] + 30, lock_start[1] + 30
        img = cv2.rectangle(img, lock_start, lock_end, color=(255, 255, 255), thickness=-1)

    res['door_locked'] = door_locked
    res['camera_blocked'] = False
    res['door_open'] = False
    return img, res


def draw_person(img, res):
    head_radius = 6
    offsets = np.random.randint(-10, 10, size=4)
    head = int(30 + offsets[0]), int(10 + offsets[1])

    img = cv2.circle(img, head, head_radius, color=(0, 255, 0), thickness=-1)

    res['person_present'] = True
    res['face_x1'] = head[0] - head_radius
    res['face_y1'] = head[1] - head_radius
    res['face_w'] = 2 * head_radius
    res['face_h'] = 2 * head_radius

    offsets = np.random.randint(-2, 2, size=4)
    d = head_radius * 2
    rec_start = head[0] - head_radius + offsets[0], head[1] + head_radius + offsets[1]
    rec_end = rec_start[0] + d + offsets[2], rec_start[1] + 30 + offsets[3]
    cv2.rectangle(img, rec_start, rec_end, color=(0, 0, 255), thickness=-1)

    res['body_x1'] = rec_start[0]
    res['body_y1'] = rec_start[1]
    res['body_w'] = rec_end[0] - rec_start[0]
    res['body_h'] = rec_end[1] - rec_start[1]

    return img, res


def generate_image_with_person():
    img, res = generate_door_open_image()
    img, res = draw_person(img, res)
    return img, res


def generate_sample():
    generators = [generate_camera_blocked_image,
                  generate_door_open_image,
                  partial(generate_door_closed_image, door_locked=True),
                  partial(generate_door_closed_image, door_locked=False),
                  generate_image_with_person]
    choice = np.random.randint(0, len(generators), size=1)[0]
    return generators[choice]()


def create_df_and_images_tensor():
    imgs = []
    rows = []
    names = []
    for i in range(int(1e4)):
        img, row = generate_sample()
        imgs.append(torch.tensor(img).permute(2, 0, 1))
        rows.append(row)
        names.append(f'{i}.jpg')

    df = pd.DataFrame(rows)
    df['img'] = names
    df.loc[:5, 'camera_blocked'] = np.nan
    return torch.stack(imgs, dim=0).float() / 255., df


def synthenic_dataset_preparation():
    def bounded_regression_converter(values):
        values = values.astype(float) / 64
        values[np.isnan(values)] = -1
        return torch.tensor(values).float().unsqueeze(dim=-1)

    class BoundedRegressionDecoder(Decoder):
        def __call__(self, x):
            return x * 64

        def tune(self, predictions, targets):
            return {}

        def load_tuned(self, params):
            pass

    def bounded_regression_task(name, labels):
        return BoundedRegressionTask(name, labels, module=nn.Linear(256, 1), decoder=BoundedRegressionDecoder())

    def binary_classification_task(name, labels):
        return BinaryClassificationTask(name, labels, module=nn.Linear(256, 1))

    imgs, df = create_df_and_images_tensor()
    output_col = ['camera_blocked', 'door_open', 'person_present', 'door_locked',
                  'face_x1', 'face_y1', 'face_w', 'face_h',
                  'body_x1', 'body_y1', 'body_w', 'body_h']
    type_guesser = TypeGuesser()
    type_guesser.type_mapping['camera_blocked'] = 'binary'
    type_guesser.type_mapping['door_open'] = 'binary'
    type_guesser.type_mapping['person_present'] = 'binary'
    type_guesser.type_mapping['door_locked'] = 'binary'
    type_guesser.type_mapping['face_x1'] = 'continuous'
    type_guesser.type_mapping['face_y1'] = 'continuous'
    type_guesser.type_mapping['face_w'] = 'continuous'
    type_guesser.type_mapping['face_h'] = 'continuous'
    type_guesser.type_mapping['body_x1'] = 'continuous'
    type_guesser.type_mapping['body_y1'] = 'continuous'
    type_guesser.type_mapping['body_w'] = 'continuous'
    type_guesser.type_mapping['body_h'] = 'continuous'
    type_guesser.type_mapping['img'] = 'img'
    values_converter = ValuesConverter()
    values_converter.type_mapping['img'] = lambda x: imgs
    values_converter.type_mapping['binary'] = binary_value_converter
    values_converter.type_mapping['continuous'] = bounded_regression_converter

    task_converter = TaskConverter()

    task_converter.type_mapping['binary'] = binary_classification_task
    task_converter.type_mapping['continuous'] = bounded_regression_task

    converters = Converters()
    converters.task = task_converter
    converters.type = type_guesser
    converters.values = values_converter

    project = Project(df, input_col='img', output_col=output_col, converters=converters, project_dir='./security_project')

    @project.add_flow
    def face_regression(flow, x, out):
        out += flow.face_x1(x.features)
        out += flow.face_y1(x.features)
        out += flow.face_w(x.features)
        out += flow.face_h(x.features)
        return out

    @project.add_flow
    def body_regression(flow, x, out):
        out += flow.body_x1(x.features)
        out += flow.body_y1(x.features)
        out += flow.body_w(x.features)
        out += flow.body_h(x.features)
        return out

    @project.add_flow
    def person_regression(flow, x, out):
        out += flow.face_regression(x)
        out += flow.body_regression(x)
        return out

    @project.add_flow
    def full_flow(flow, x, out):
        out += flow.camera_blocked(x.features)
        out += flow.door_open(x.features) | (~out.camera_blocked)
        out += flow.door_locked(x.features) | (~out.door_open)
        out += flow.person_present(x.features) | out.door_open
        out += flow.person_regression(x) | out.person_present
        return out

    dataset = project.get_full_flow().get_dataset()
    train_dataset, val_dataset = torch_split_dataset(dataset, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32 * torch.cuda.device_count(), shuffle=False)
    nested_loaders = OrderedDict({
        'train': train_loader,
        'valid': val_loader
    })

    class SecurityModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.Conv2d(64, 128, kernel_size=5),
                nn.AvgPool2d(2),
                nn.Conv2d(128, 128, kernel_size=5),
                nn.Conv2d(128, 256, kernel_size=5),
                nn.AvgPool2d(2),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.flow_module = full_flow.torch()

        def forward(self, x):
            res = {}
            res['features'] = self.seq(x['img'])
            res['gt'] = x.get('gt')
            return self.flow_module(res)

    model = SecurityModule()
    datasets = {
        'train': train_dataset,
        'valid': val_dataset,
        'infer': val_dataset
    }
    return model, nested_loaders, datasets, project
