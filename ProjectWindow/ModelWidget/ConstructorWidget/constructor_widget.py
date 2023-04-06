from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer

from ProjectWindow.ModelWidget.ConstructorWidget.CanvasWidget.canvas_widget import CanvasWidget
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer, InputLayer, OutputLayer
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.linear_layers import Identity, Linear

from loguru import logger
from typing import Type


class ConstructorMenu(MenuWidget):
    def __init__(self, config: Config):
        super(ConstructorMenu, self).__init__(config)
        self.create_identity_button = QPushButton('Identity')
        self.create_linear_button = QPushButton('Linear')

        self.layer_menu_container = MenuContainer()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.create_identity_button)
        self.layout().addWidget(self.create_linear_button)
        self.layout().addWidget(self.layer_menu_container)


class ConstructorWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ConstructorWidget, self).__init__(menu_container, config, ConstructorMenu)
        self.id = 2
        self.canvas = CanvasWidget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.input_layer = InputLayer(self.menu.layer_menu_container, config, self.canvas.canvas, 0,
                                      pos=QPoint(50, 10))
        self.input_layer.out_button.clicked.connect(lambda: self.out_button_clicked(self.input_layer))
        self.input_layer.newGeometry.connect(lambda x: self.canvas.canvas.update_arrows())
        self.output_layer = OutputLayer(self.menu.layer_menu_container, config, self.canvas.canvas, 1,
                                        pos=QPoint(50, 300))
        self.output_layer.in_button.clicked.connect(lambda: self.in_button_clicked(self.output_layer))
        self.output_layer.newGeometry.connect(lambda x: self.canvas.canvas.update_arrows())
        self.canvas.canvas.layers = [self.input_layer, self.output_layer]

        self.menu.create_identity_button.clicked.connect(lambda: self.create_layer(Identity))
        self.menu.create_linear_button.clicked.connect(lambda: self.create_layer(Linear))

    def create_layer(self, layer: Type[Layer]):
        layer = layer(self.menu.layer_menu_container, self.config, self.canvas.canvas, self.id)
        layer.in_button.clicked.connect(lambda: self.in_button_clicked(layer))
        layer.out_button.clicked.connect(lambda: self.out_button_clicked(layer))
        layer.newGeometry.connect(lambda x: self.canvas.canvas.update_arrows())
        self.id += 1
        self.canvas.canvas.layers.append(layer)
        self.canvas.update()
        logger.info(f'Created layer')

    def in_button_clicked(self, layer):
        existing_arrow = None
        for arrow in self.canvas.canvas.arrows:
            if arrow[1] == layer:
                existing_arrow = arrow
        if self.canvas.canvas.new_arrow is None:
            self.canvas.canvas.current_arrow = existing_arrow
        else:
            if existing_arrow is not None:
                self.canvas.canvas.arrows.remove(existing_arrow)
            current_arrow = (self.canvas.canvas.new_arrow, layer)
            self.canvas.canvas.arrows.append(current_arrow)
            self.canvas.canvas.new_arrow = None
            self.canvas.canvas.current_arrow = current_arrow
        self.canvas.canvas.update_arrows()

    def out_button_clicked(self, layer):
        self.canvas.canvas.new_arrow = layer
