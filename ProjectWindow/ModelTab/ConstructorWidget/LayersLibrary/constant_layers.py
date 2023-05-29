from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config

from loguru import logger
import torch.nn as nn
import torch


class nnEyeLayer(nn.Module):
    def __init__(self, n):
        super(nnEyeLayer, self).__init__()
        self.out = torch.eye(n).detach()

    def forward(self):
        return self.out


class EyeLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Возвращает единичную матрицу заданного размера'

    def parameters(self):
        n = QSpinBox()
        n.setMaximum(100000)
        n.setMinimum(1)

        self.params = {
            'n': n
        }


class EyeLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(EyeLayer, self).__init__(menu_container, config, parent, id, nnEyeLayer, EyeLayerMenu, pos,
                                          name='Eye', color=QColor(196, 196, 196), in_buttons=[])

    def forward(self, x):
        return {'out': self.F()}


class nnIntLayer(nn.Module):
    def __init__(self, n):
        super(nnIntLayer, self).__init__()
        self.out = n

    def forward(self):
        return self.out


class IntLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Возвращает заданное целое число'

    def parameters(self):
        n = QSpinBox()
        n.setMaximum(100000)
        n.setMinimum(-100000)

        self.params = {
            'n': n
        }


class IntLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(IntLayer, self).__init__(menu_container, config, parent, id, nnIntLayer, IntLayerMenu, pos,
                                          name='Int', color=QColor(196, 196, 196), in_buttons=[])

    def forward(self, x):
        return {'out': self.F()}