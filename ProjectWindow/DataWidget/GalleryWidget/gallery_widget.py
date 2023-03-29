from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget
from ProjectWindow.DataWidget.data_handler import JsonHandler

import os

from loguru import logger


class GalleryMenu(MenuWidget):
    def __init__(self, config: Config):
        super(GalleryMenu, self).__init__(config)
        self.json_handler = None

        self.update_disk = QPushButton('Обновить (не работает)')
        self.merge_labels = QPushButton('Объединить разметку (не работает)')

        self.name = QLabel('')

        self.labels = QListWidget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.update_disk)
        self.layout().addWidget(self.merge_labels)
        self.layout().addWidget(self.name)
        self.layout().addWidget(self.labels)

    def set_json_handler(self, json_handler: JsonHandler):
        self.json_handler = json_handler

    def update_image(self):
        self.name.setText(self.json_handler.current_image)
        self.labels.clear()
        self.labels.addItems([', '.join([f'{k}: {round(v, 3)}' for k, v in r.items()])
                              for r in self.json_handler.labels])


class GalleryItem(QWidget):
    def __init__(self, image_path: str, image_name: str, json_handler: JsonHandler):
        super(GalleryItem, self).__init__()
        self.image_path = image_path
        self.image_name = image_name
        self.loaded = False

        self.setFixedSize(200, 240)

        self.icon = QPushButton()
        self.icon.setFixedSize(200, 200)
        self.icon.clicked.connect(lambda: json_handler.change_image(self.image_name))

        self.name = QLabel(image_name)
        self.name.setFixedSize(200, 40)
        self.name.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.icon)
        self.layout().addWidget(self.name)

    def load_image(self):
        logger.debug(f'Loading image {self.image_path}')
        self.icon.setIcon(QIcon(self.image_path))
        self.icon.setIconSize(QSize(200, 200))
        self.loaded = True

    def clear_image(self):
        self.icon.setIcon(QIcon())
        self.loaded = False


class GalleryWidget(WidgetWithMenu):
    image_chosen = pyqtSignal()

    def __init__(self, menu_container: MenuContainer, config: Config, json_handler: JsonHandler):
        super().__init__(menu_container, config, GalleryMenu)
        logger.info('Initializing GalleryWidget')
        self.menu.set_json_handler(json_handler)

        self.setLayout(QHBoxLayout())

        self.json_handler = json_handler

        self.list_of_images = []
        self.items = []

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.horizontalScrollBar().valueChanged.connect(self.update_visible_images)
        self.scrollArea.verticalScrollBar().valueChanged.connect(self.update_visible_images)

        self.scrollAreaWidgetContents = QWidget()

        self.images = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.layout().addWidget(self.scrollArea)

        self.load_images()
        self.update_visible_images()

    def load_images(self):
        for row, image in enumerate(self.json_handler.data.keys()):
            self.add_list_item(image, row)

    def add_list_item(self, image, row):
        image_path = os.path.join(self.config.data_path, image)

        item = GalleryItem(image_path, image, self.json_handler)

        self.images.addWidget(item, row // 5, row % 5, alignment=Qt.AlignmentFlag.AlignRight)
        self.items.append(item)

    def update_image(self, image_name):
        self.json_handler.change_image(image_name)
        self.json_handler.current_image = image_name
        self.image_chosen.emit()

    def update_visible_images(self):
        for image in self.items:
            if image.visibleRegion().isEmpty() and image.loaded:
                image.clear_image()
            elif not image.visibleRegion().isEmpty() and not image.loaded:
                image.load_image()
