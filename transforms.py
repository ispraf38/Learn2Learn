import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    def __init__(self, transforms=A.Compose([ToTensorV2()])):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        new_image = self.transforms(image=np.array(image))['image']
        return new_image