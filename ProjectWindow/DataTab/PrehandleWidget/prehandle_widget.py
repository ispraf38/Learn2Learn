from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget
from ProjectWindow.DataTab.PrehandleWidget.PrehandleLibrary.albumentations import *
from ProjectWindow.DataTab.PrehandleWidget.PrehandleLibrary.base_layer import PrehandleLayer
from ProjectWindow.DataTab.MainWidget.datasets import Transforms

import os
import numpy as np
import cv2
from loguru import logger
from functools import partial
from typing import Type, Dict, Union, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


BUTTONS = {
    'Добавить слой предобработки': {
        'InvertImg': AInvertImg,
        'Equalize': AEqualize,
        'CLAHE': ACLAHE,
        'ToSepia': AToSepia,
        'GaussianBlur': AGaussianBlur,
        'HueSaturationValue': AHueSaturationValue,
        'RandomContrast': ARandomContrast,
        'Resize': AResize
    },
    'Удалить слой': 'delete layer'
}


FONT = QFont('Ariel', 16)


class PrehandleLayerMenu(QWidget):
    create_layer = pyqtSignal(type(PrehandleLayer))

    def __init__(self):
        super(PrehandleLayerMenu, self).__init__()
        self.setLayout(QStackedLayout())
        self.delete_layer = None
        self.layouts = {}
        self.buttons = {}
        self.add_layout(BUTTONS, 'main')
        logger.debug(f'Create layer menu: {self.layouts}')
        self.set_widget('main')

    def add_layout(self, buttons: Dict[str, Union[str, Dict, Type[PrehandleLayer]]], key: str, back: bool = False) \
            -> (QWidget, Optional[QPushButton]):
        layout = QVBoxLayout()
        back_buttons = []
        for k, v in buttons.items():
            button = QPushButton(k)
            button.setFont(FONT)
            if type(v) == dict:
                widget, bb = self.add_layout(v, k, True)
                back_buttons.append(bb)
                logger.debug(f'Connecting {k} button')
                button.clicked.connect(partial(self.set_widget, k))
            elif type(v) == str:
                if v == 'delete layer':
                    self.delete_layer = button
            else:
                button.clicked.connect(partial(self.create_layer.emit, v))
            layout.addWidget(button)

        if back:
            back_button = QPushButton('Назад')
            back_button.setFont(FONT)
            layout.addWidget(back_button)
        else:
            back_button = None

        for bb in back_buttons:
            bb.clicked.connect(lambda: self.set_widget(key))

        widget = QWidget()
        widget.setLayout(layout)

        self.layouts[key] = widget
        return widget, back_button

    def set_widget(self, name: str):
        logger.debug(f'Pressed {name} button')
        self.layout().addWidget(self.layouts[name])
        self.layout().setCurrentWidget(self.layouts[name])


class ImageWidget(QWidget):
    def __init__(self, config: Config, parent_widget: QWidget):
        super().__init__()
        self.config = config
        self.parent_widget = parent_widget

        self.label = QLabel()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.image_width = None
        self.image_height = None
        self.img = None

    def set_image(self, image_path):
        screen_size = self.parent_widget.size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()

        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        if len(self.img.shape) == 3:
            height, width, bytesPerComponent = self.img.shape
            format = QImage.Format.Format_RGB888
            bytesPerLine = 3 * width
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        elif len(self.img.shape) == 2:
            height, width = self.img.shape
            format = QImage.Format.Format_MonoLSB
            bytesPerLine = width
        else:
            raise TypeError
        qimg = QImage(self.img.data, width, height, bytesPerLine, format)

        self.pixmap = QPixmap.fromImage(qimg).scaledToHeight(int(screen_height * 0.9))

        if self.pixmap.width() > int(1. * screen_width):
            self.pixmap = QPixmap.fromImage(qimg).scaledToWidth(int(screen_width * 0.9))

        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(self.pixmap.size())
        self.image_width, self.image_height = self.pixmap.width(), self.pixmap.height()


class PrehandleMenu(MenuWidget):
    def __init__(self, config: Config):
        super(PrehandleMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        self.layer_menu = PrehandleLayerMenu()
        self.layout().addWidget(self.layer_menu)

        self.layer_menu_container = MenuContainer()
        self.layout().addWidget(self.layer_menu_container)


class PrehandleLayerContainer(QWidget):
    def __init__(self, config: Config, menu_container: MenuContainer):
        super(PrehandleLayerContainer, self).__init__()
        self.config = config
        self.menu_container = menu_container
        self.setLayout(QVBoxLayout())
        self.layers = []
        self.current_layer = None

    def add_layer(self, layer: Type[PrehandleLayer]):
        layer = layer(self.menu_container, self.config)
        self.layers.append(layer)
        self.current_layer = layer
        layer.activate_menu()
        layer.button.clicked.connect(partial(self.set_current_layer, layer))
        self.rebuild_layout()

    def delete_current_layer(self):
        if self.current_layer is not None:
            self.layers.remove(self.current_layer)
            self.current_layer.close()
            self.current_layer = None
            self.rebuild_layout()

    def rebuild_layout(self):
        self.setLayout(QHBoxLayout())
        for layer in self.layers:
            self.layout().addWidget(layer)

    def set_current_layer(self, layer: PrehandleLayer):
        self.current_layer = layer

    def gen_transform(self, to_tensor=True):
        transforms = [layer.transform(**layer.get_params()) for layer in self.layers]
        if to_tensor:
            transforms += [ToTensorV2()]

        return Transforms(A.Compose(transforms))


class PrehandleWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(PrehandleWidget, self).__init__(menu_container, config, PrehandleMenu)

        self.image_path = config.prehandle_default_test_image
        self.transform = None

        left_widget = QWidget()
        left_widget.setLayout(QVBoxLayout())
        right_widget = QWidget()
        right_widget.setLayout(QVBoxLayout())
        middle_widget = QWidget()
        middle_widget.setLayout(QVBoxLayout())
        middle_widget.setFixedWidth(300)

        self.left_button = QPushButton('Выбрать изображение')
        self.left_button.setFont(FONT)
        self.left_button.clicked.connect(self.choose_image)
        self.right_button = QPushButton('Обновить')
        self.right_button.setFont(FONT)
        self.right_button.clicked.connect(partial(self.update_image, None))

        self.left_image = ImageWidget(config, left_widget)
        self.right_image = ImageWidget(config, right_widget)

        left_widget.layout().addWidget(self.left_image)
        left_widget.layout().addWidget(self.left_button)

        right_widget.layout().addWidget(self.right_image)
        right_widget.layout().addWidget(self.right_button)

        self.middle_container = PrehandleLayerContainer(config, self.menu.layer_menu_container)
        self.menu.layer_menu.create_layer.connect(lambda x: self.middle_container.add_layer(x))
        self.menu.layer_menu.delete_layer.clicked.connect(self.middle_container.delete_current_layer)

        self.save_button = QPushButton('Сохранить')
        self.save_button.setFont(FONT)
        self.save_button.clicked.connect(self.gen_transform)

        middle_widget.layout().addWidget(self.middle_container)
        middle_widget.layout().addWidget(self.save_button)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(left_widget)
        self.layout().addWidget(middle_widget)
        self.layout().addWidget(right_widget)

        self.update_image(self.image_path)

    def gen_transform(self):
        self.transform = self.middle_container.gen_transform()

    def update_image(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        self.left_image.set_image(image_path)
        self.gen_transform()
        transform = self.middle_container.gen_transform(False)
        cv2.imwrite('temp.jpg', cv2.cvtColor(transform(self.left_image.img), cv2.COLOR_RGB2BGR))

        self.right_image.set_image('temp.jpg')

    def choose_image(self):
        self.image_path = QFileDialog.getOpenFileName(self, 'Open Image', self.config.project_path,
                                                      'Image Files (*.png *.jpg *.bmp)')[0]
        self.update_image()
