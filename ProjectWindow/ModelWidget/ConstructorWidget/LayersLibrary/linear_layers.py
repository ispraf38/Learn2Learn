from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import torch.nn as nn


class IdentityMenu(LayerMenu):
    def description(self):
        return 'Слой - пустышка, возвращает то же, что получает на вход'


class IdentityLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(IdentityLayer, self).__init__(menu_container, config, parent, id, nn.Identity, IdentityMenu, pos,
                                            name='Identity', color=QColor(196, 196, 196))


class LinearMenu(LayerMenu):
    def description(self):
        return 'Применяет линейное преобразование ко входныи данным'

    def parameters(self):
        in_features = QSpinBox()
        in_features.setMaximum(999999)
        in_features.setMinimum(1)

        out_features = QSpinBox()
        out_features.setMaximum(999999)
        out_features.setMinimum(1)

        bias = QCheckBox()
        bias.setChecked(True)
        self.params = {
            'in_features': in_features,
            'out_features': out_features,
            'bias': bias
        }


class LinearLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(LinearLayer, self).__init__(menu_container, config, parent, id, nn.Linear, LinearMenu, pos,
                                          name='Linear', color=QColor(96, 255, 96))

