from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer

from ProjectWindow.ModelWidget.ConstructorWidget.CanvasWidget.canvas_widget import CanvasWidget
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer, InputLayer, OutputLayer
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.activation_layers import *
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.convolution_layers import *
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.dropout_layers import *
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.linear_layers import *
from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.utility_layers import *

from loguru import logger
from typing import Type, Dict, Union, Optional, Tuple
from functools import partial


BUTTONS = {
    'Создать слой': {
        'Activation layers': {
            'ReLU': ReLULayer,
            'Sigmoid': SigmoidLayer
        },
        'Convolution layers': {
            'Conv1d': Conv1dLayer,
            'Conv2d': Conv2dLayer
        },
        'Dropout layers': {
            'Dropout': DropoutLayer
        },
        'Linear layers': {
            'Identity': IdentityLayer,
            'Linear': LinearLayer
        },
        'Utility layers': {
            'Flatten': FlattenLayer
        }
    },
    'Удалить выбранный слой': 'delete layer',
    'Удалить выбранную связь': 'delete connection'
}


class CreateLayerMenu(QWidget):
    create_layer = pyqtSignal(type(Layer))

    def __init__(self):
        super(CreateLayerMenu, self).__init__()
        self.setLayout(QStackedLayout())
        self.delete_layer = None
        self.delete_connection = None
        self.layouts = {}
        self.buttons = {}
        self.add_layout(BUTTONS, 'main')
        logger.debug(f'Create layer menu: {self.layouts}')
        self.set_widget('main')

    def add_layout(self, buttons: Dict[str, Union[str, Dict, Type[Layer]]], key: str, back: bool = False)\
            -> (QWidget, Optional[QPushButton]):
        layout = QVBoxLayout()
        back_buttons = []
        for k, v in buttons.items():
            button = QPushButton(k)
            if type(v) == dict:
                widget, bb = self.add_layout(v, k, True)
                back_buttons.append(bb)
                logger.debug(f'Connecting {k} button')
                button.clicked.connect(partial(self.set_widget, k))
            elif type(v) == str:
                if v == 'delete layer':
                    self.delete_layer = button
                elif v == 'delete connection':
                    self.delete_connection = button
            else:
                button.clicked.connect(partial(self.create_layer.emit, v))
            layout.addWidget(button)

        if back:
            back_button = QPushButton('Назад')
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


class ConstructorMenu(MenuWidget):
    def __init__(self, config: Config):
        super(ConstructorMenu, self).__init__(config)
        self.create_layer_menu = CreateLayerMenu()

        self.layer_menu_container = MenuContainer()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.create_layer_menu)
        self.layout().addWidget(self.layer_menu_container)


class ConstructorWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ConstructorWidget, self).__init__(menu_container, config, ConstructorMenu)
        self.id = 2
        self.canvas = CanvasWidget()
        self.canvas_ = self.canvas.canvas

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.input_layer = InputLayer(self.menu.layer_menu_container, config, self.canvas_, 0,
                                      pos=QPoint(50, 10))
        self.input_layer.out_button.clicked.connect(lambda: self.canvas_.out_button_clicked(self.input_layer))
        self.input_layer.newGeometry.connect(lambda x: self.canvas_.update_arrows())
        self.input_layer.chosen.connect(lambda: self.canvas_.set_current_layer(None))

        self.output_layer = OutputLayer(self.menu.layer_menu_container, config, self.canvas_, 1,
                                        pos=QPoint(50, 300))
        self.output_layer.in_button.clicked.connect(lambda: self.canvas_.in_button_clicked(self.output_layer))
        self.output_layer.newGeometry.connect(lambda x: self.canvas_.update_arrows())
        self.output_layer.chosen.connect(lambda: self.canvas_.set_current_layer(None))

        self.canvas_.layers = [self.input_layer, self.output_layer]

        self.menu.create_layer_menu.create_layer.connect(lambda x: self.create_layer(x))
        self.menu.create_layer_menu.delete_connection.clicked.connect(self.canvas_.delete_current_connection)
        self.menu.create_layer_menu.delete_layer.clicked.connect(self.canvas_.delete_current_layer)

    def create_layer(self, layer: Type[Layer]):
        layer = layer(self.menu.layer_menu_container, self.config, self.canvas_, self.id)
        layer.in_button.clicked.connect(lambda: self.canvas_.in_button_clicked(layer))
        layer.out_button.clicked.connect(lambda: self.canvas_.out_button_clicked(layer))
        layer.newGeometry.connect(lambda x: self.canvas_.update_arrows())
        layer.chosen.connect(lambda: self.canvas_.set_current_layer(layer))
        self.id += 1
        self.canvas_.layers.append(layer)
        # self.canvas_.set_current_layer(layer)
        self.canvas.update()
        logger.info(f'Created layer {layer}')
