from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import json
import os.path

from ProjectWindow.utils import Config

from loguru import logger


class JsonHandler(QObject):
    image_changed = pyqtSignal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.current_image = None
        self.data = {}
        self.order = [None]
        self.load()
        self.update_images()

    @property
    def labels(self):
        if self.current_image is None:
            return []
        labels = self.data.get(self.current_image).get('labels')
        if labels is None:
            return []
        return labels

    def update_images(self):
        list_of_images = os.listdir(self.config.data_path)
        list_of_images = filter(lambda f: f.lower().endswith(('.jpg', '.jpeg', '.png')), list_of_images)
        list_of_images = sorted(list_of_images)
        for image in list_of_images:
            if image not in self.data:
                self.data[image] = {
                    'labels': [],
                    'handled': False
                }
        self.save()

    def load(self):
        logger.info('Loading json_handler')
        if os.path.exists(self.config.json_path):
            with open(self.config.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {}
            self.save()
        self._get_order()

    def save(self):
        logger.info('Saving json_handler')
        with open(self.config.json_path, 'w+', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def next(self):
        self.change_image(
            self.order[self.order.index(self.current_image) + 1]
        )
        self.save()

    def previous(self):
        self.change_image(
            self.order[self.order.index(self.current_image) - 1]
        )
        self.save()

    def _get_order(self):
        for k in self.data.keys():
            self.order.append(k)
        self.order.append(None)

    def change_image(self, image):
        if image is not None:
            self.current_image = image
            self.image_changed.emit()

    def clear_labels(self):
        if self.current_image is not None:
            self.data[self.current_image]['labels'].clear()
