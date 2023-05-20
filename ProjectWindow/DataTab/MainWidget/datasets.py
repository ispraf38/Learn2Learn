from torch.utils.data import Dataset

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget

import pandas as pd
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from loguru import logger

import torch
import torch.nn as nn


class Transforms:
    def __init__(self, transforms=A.Compose([ToTensorV2()])):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        new_image = self.transforms(image=np.array(image))['image']
        return new_image


class DefaultDataset(Dataset):
    def __init__(self, config: Config, transform: Transforms, *args, **kwargs):
        super(DefaultDataset, self).__init__(*args, **kwargs)
        self.config = config
        self.transform = transform

    def __getitem__(self, item):
        pass

    def __len__(self) -> int:
        pass


def get_proposal_by_label(label, img_size=32):
    cur_id = 0
    best_id = -1
    min_loss = 10000
    deltas = None
    num_props = img_size - 5 + 1
    for i in range(0, num_props, 2):
        for j in range(0, num_props, 2):
            l = j / img_size
            r = (j + 5) / img_size
            t = i / img_size
            b = (i + 5) / img_size
            loss = nn.functional.mse_loss(torch.Tensor([l, r, t, b]), torch.Tensor([*label.values()]))
            if loss < min_loss:
                best_id = cur_id
                min_loss = loss
                deltas = [x - y for (x, y) in zip([l, r, t, b], [label['l'], label['r'], label['t'], label['b']])]
            cur_id += 1
    return best_id, deltas


def handle_labels(labels, img_size=32):
    new_labels = [[0, 0, 0, 0, 0] for _ in range(((img_size - 5 + 1) // 2) ** 2)]
    for label in labels:
        best_id, deltas = get_proposal_by_label(label, img_size)
        new_labels[best_id] = deltas + [1]

    return torch.Tensor(new_labels)


class ImageObjectDetectionDataset(DefaultDataset):
    def __init__(self, config: Config, transform: Transforms):
        super(ImageObjectDetectionDataset, self).__init__(config, transform)
        self.config = config
        with open(self.config.json_path, 'r', encoding='utf-8') as f:
            self.json = json.load(f)
        # self.json = pd.read_json(self.config.json_path)
        self.sample = [(k, handle_labels(v['labels'])) for (k, v) in self.json.items()]

    def __getitem__(self, item):
        image_path, labels = self.sample[item]
        stream = open(os.path.join(self.config.data_path, image_path), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        image = self.transform(image)
        return image, labels

    def __len__(self) -> int:
        return len(self.json)
