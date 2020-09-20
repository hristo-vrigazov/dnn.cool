from collections import OrderedDict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dnn_cool.converters import TypeGuesser, ValuesConverter, TaskConverter, Converters
from dnn_cool.decoders import BoundedRegressionDecoder
from dnn_cool.project import Project
from dnn_cool.task_converters import To
from dnn_cool.task_flow import BoundedRegressionTask, BinaryClassificationTask, ClassificationTask, \
    MultilabelClassificationTask
from dnn_cool.utils import torch_split_dataset
from dnn_cool.value_converters import binary_value_converter, classification_converter, \
    ImageCoordinatesValuesConverter, MultiLabelValuesConverter


def add_random_noise(img):
    n = 40
    i = np.random.randint(0, 64, size=n)
    j = np.random.randint(0, 64, size=n)
    c = np.random.randint(0, 3, size=n)
    values = np.random.randint(0, 255, size=n)
    img[i, j, c] = values


def generate_camera_blocked_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # randomly add some noise
    add_random_noise(img)
    return img, {'camera_blocked': True}


def generate_door_open_image():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    # randomly add some noise
    add_random_noise(img)
    return img, {'camera_blocked': False, 'door_open': True, 'person_present': False}


def generate_door_closed_image(door_locked):
    img, res = generate_door_open_image()

    offsets = np.random.randint(-10, 10, size=4)
    x1, y1 = int(10 + offsets[0]), int(10 + offsets[1])
    x2, y2 = int(50 + offsets[2]), int(50 + offsets[2])

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(139, 69, 19), thickness=-1)

    if door_locked:
        lock_start = x1, int((y1 + y2) / 2)
        lock_end = lock_start[0] + 10, lock_start[1] + 10
        img = cv2.rectangle(img, lock_start, lock_end, color=(0, 0, 255), thickness=-1)

    res['door_locked'] = door_locked
    res['camera_blocked'] = False
    res['door_open'] = False
    return img, res


def select_facial_characteristics():
    choices = [
        [(255, 0, 255), (0, 2)],
        [(0, 255, 255), (1, 2)],
        [(255, 255, 0), (0, 1)],
        [(0, 0, 0), ()],
        [(255, 0, 0), (0,)],
        [(0, 255, 0), (1,)],
        [(0, 0, 255), (2,)],
    ]
    choice = np.random.randint(0, len(choices), size=1)[0]
    return choices[choice]


def draw_person(img, res, shirt_type='blue'):
    head_radius = 6
    offsets = np.random.randint(-4, 4, size=4)
    head = int(30 + offsets[0]), int(10 + offsets[1])

    color, face_characteristics = select_facial_characteristics()
    img = cv2.circle(img, head, head_radius, color=color, thickness=-1)

    res['person_present'] = True
    res['face_x1'] = head[0] - head_radius
    res['face_y1'] = head[1] - head_radius
    res['face_w'] = 2 * head_radius
    res['face_h'] = 2 * head_radius
    res['facial_characteristics'] = ','.join(map(str, face_characteristics))

    offsets = np.random.randint(-2, 2, size=4)
    d = head_radius * 2
    rec_start = head[0] - head_radius + offsets[0], head[1] + head_radius + offsets[1]
    rec_end = rec_start[0] + d + offsets[2], rec_start[1] + 30 + offsets[3]

    if shirt_type == 'blue':
        color = (0, 0, 255)
        shirt_label = 0
    elif shirt_type == 'red':
        color = (255, 0, 0)
        shirt_label = 1
    elif shirt_type == 'yellow':
        color = (255, 255, 0)
        shirt_label = 2
    elif shirt_type == 'cyan':
        color = (0, 255, 255)
        shirt_label = 3
    elif shirt_type == 'magenta':
        color = (255, 0, 255)
        shirt_label = 4
    elif shirt_type == 'green':
        color = (0, 255, 0)
        shirt_label = 5
    else:
        # black
        color = (0, 0, 0)
        shirt_label = 6

    cv2.rectangle(img, rec_start, rec_end, color=color, thickness=-1)

    res['body_x1'] = rec_start[0]
    res['body_y1'] = rec_start[1]
    res['body_w'] = rec_end[0] - rec_start[0]
    res['body_h'] = rec_end[1] - rec_start[1]
    res['shirt_type'] = shirt_label

    return img, res


def generate_image_with_person(shirt_type='blue'):
    img, res = generate_door_open_image()
    img, res = draw_person(img, res, shirt_type)
    return img, res


def generate_sample():
    generators = [generate_camera_blocked_image,
                  generate_door_open_image,
                  partial(generate_door_closed_image, door_locked=True),
                  partial(generate_door_closed_image, door_locked=False),
                  partial(generate_image_with_person, shirt_type='blue'),
                  partial(generate_image_with_person, shirt_type='red'),
                  partial(generate_image_with_person, shirt_type='yellow'),
                  partial(generate_image_with_person, shirt_type='cyan'),
                  partial(generate_image_with_person, shirt_type='magenta'),
                  partial(generate_image_with_person, shirt_type='green'),
                  partial(generate_image_with_person, shirt_type='black')
                  ]
    choice = np.random.randint(0, len(generators), size=1)[0]
    return generators[choice]()


def create_df_and_images_tensor(n=int(1e4), cache_file=Path('dnn_cool_synthetic_dataset.pkl')):
    if cache_file.exists():
        return torch.load(cache_file)
    imgs = []
    rows = []
    names = []
    for i in range(n):
        img, row = generate_sample()
        imgs.append(torch.tensor(img).permute(2, 0, 1))
        rows.append(row)
        names.append(f'{i}.jpg')

    df = pd.DataFrame(rows)
    df['syn_img'] = names
    df.loc[:5, 'camera_blocked'] = np.nan
    res = torch.stack(imgs, dim=0).float() / 255., df
    torch.save(res, cache_file)
    return res


def synthetic_dataset_preparation(n=int(1e4)):
    imgs, df = create_df_and_images_tensor(n)
    output_col = ['camera_blocked', 'door_open', 'person_present', 'door_locked',
                  'face_x1', 'face_y1', 'face_w', 'face_h',
                  'facial_characteristics',
                  'body_x1', 'body_y1', 'body_w', 'body_h', 'shirt_type']
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
    type_guesser.type_mapping['syn_img'] = 'img'
    type_guesser.type_mapping['shirt_type'] = 'category'
    type_guesser.type_mapping['facial_characteristics'] = 'multilabel'

    values_converter = ValuesConverter()
    values_converter.type_mapping['img'] = lambda x: imgs
    values_converter.type_mapping['binary'] = binary_value_converter
    values_converter.type_mapping['continuous'] = ImageCoordinatesValuesConverter(dim=64)
    values_converter.type_mapping['category'] = classification_converter

    multilabel_converter = MultiLabelValuesConverter()
    values_converter.type_mapping['multilabel'] = multilabel_converter

    task_converter = TaskConverter()

    task_converter.type_mapping['binary'] = To(BinaryClassificationTask,
                                               module_supplier=partial(nn.Linear, in_features=256, out_features=1))
    task_converter.type_mapping['continuous'] = To(BoundedRegressionTask,
                                                   module_supplier=partial(nn.Linear, in_features=256, out_features=1),
                                                   decoder_supplier=partial(BoundedRegressionDecoder, scale=64))
    n_classes = classification_converter(df['shirt_type']).max().item() + 1
    task_converter.type_mapping['category'] = To(ClassificationTask,
                                                 module_supplier=partial(nn.Linear,
                                                                         in_features=256, out_features=n_classes))
    n_classes = multilabel_converter(df['facial_characteristics']).shape[1]
    task_converter.type_mapping['multilabel'] = To(MultilabelClassificationTask,
                                                   module_supplier=partial(nn.Linear,
                                                                           in_features=256, out_features=n_classes))

    converters = Converters()
    converters.task = task_converter
    converters.type = type_guesser
    converters.values = values_converter

    project = Project(df, input_col='syn_img', output_col=output_col, converters=converters,
                      project_dir='./security_project')

    @project.add_flow
    def face_regression(flow, x, out):
        out += flow.face_x1(x.face_localization)
        out += flow.face_y1(x.face_localization)
        out += flow.face_w(x.face_localization)
        out += flow.face_h(x.face_localization)
        out += flow.facial_characteristics(x.features)
        return out

    @project.add_flow
    def body_regression(flow, x, out):
        out += flow.body_x1(x.body_localization)
        out += flow.body_y1(x.body_localization)
        out += flow.body_w(x.body_localization)
        out += flow.body_h(x.body_localization)
        out += flow.shirt_type(x.features)
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
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(128, 128, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=5),
                nn.AvgPool2d(2),
                nn.ReLU(inplace=True),
            )

            self.features_seq = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.face_localization_seq = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.body_localization_seq = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.flow_module = full_flow.torch()

        def forward(self, x):
            res = {}
            common = self.seq(x['syn_img'])
            res['features'] = self.features_seq(common)
            res['face_localization'] = self.face_localization_seq(common)
            res['body_localization'] = self.body_localization_seq(common)
            res['gt'] = x.get('gt')
            return self.flow_module(res)

    model = SecurityModule()
    datasets = {
        'train': train_dataset,
        'valid': val_dataset,
        'infer': val_dataset
    }
    return model, nested_loaders, datasets, project
