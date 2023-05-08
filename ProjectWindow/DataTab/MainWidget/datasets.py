from torch.utils.data import Dataset

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget

import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from loguru import logger


class Transforms:
    def __init__(self, transforms=A.Compose([ToTensorV2()])):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        logger.debug(np.array(image).shape)
        return self.transforms(image=np.array(image))['image']


class DefaultDataset(Dataset):
    def __init__(self, config: Config, transform: Transforms, *args, **kwargs):
        super(DefaultDataset, self).__init__(*args, **kwargs)
        self.config = config
        self.transform = transform

    def __getitem__(self, item):
        pass

    def __len__(self) -> int:
        pass


class ImageObjectDetectionDataset(DefaultDataset):
    def __init__(self, config: Config, transform: Transforms):
        super(ImageObjectDetectionDataset, self).__init__(config, transform)
        self.config = config
        self.json = pd.read_json(self.config.json_path)
        self.sample = [(k, v['labels']) for (k, v) in self.json.items()]

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
