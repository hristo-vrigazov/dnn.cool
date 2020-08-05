import numpy as np
import pandas as pd
import cv2
import torch

from functools import partial


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

