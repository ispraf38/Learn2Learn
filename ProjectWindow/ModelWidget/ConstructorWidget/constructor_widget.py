from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer

from ProjectWindow.ModelWidget.ConstructorWidget.CanvasWidget.canvas_widget import CanvasWidget
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer

from loguru import logger


class ConstructorMenu(MenuWidget):
    def __init__(self, config: Config):
        super(ConstructorMenu, self).__init__(config)
        self.create_layer_button = QPushButton('Создать пустой слой')

        self.layer_menu_container = MenuContainer()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.create_layer_button)
        self.layout().addWidget(self.layer_menu_container)


class ConstructorWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ConstructorWidget, self).__init__(menu_container, config, ConstructorMenu)
        self.canvas = CanvasWidget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.menu.create_layer_button.clicked.connect(self.create_layer)
        self.layers = []

    def create_layer(self):
        layer = Layer(self.menu.layer_menu_container, self.config, self.canvas.canvas)
        self.layers.append(layer)
        self.canvas.canvas.layers.append(layer)
        self.canvas.update()
        logger.info(f'Created layer')
