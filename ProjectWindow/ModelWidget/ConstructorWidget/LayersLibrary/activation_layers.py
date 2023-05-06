from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config

import torch.nn as nn


class ReLUMenu(LayerMenu):
    def description(self):
        return 'Поэлементный максимум от элемента и нуля'


class ReLULayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(ReLULayer, self).__init__(menu_container, config, parent, id, nn.ReLU, ReLUMenu, pos,
                                            name='ReLU', color=QColor(128, 128, 255))


class SigmoidMenu(LayerMenu):
    def description(self):
        return 'Поэлементный максимум от элемента и нуля'


class SigmoidLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(SigmoidLayer, self).__init__(menu_container, config, parent, id, nn.Sigmoid, SigmoidMenu, pos,
                                            name='Sigmoid', color=QColor(128, 128, 255))