from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import torch.nn as nn


class IdentityMenu(LayerMenu):
    def get_parameters_widget(self):
        label = QLabel('Слой - пустышка, возвращает то же, что получает на вход')
        return label


class Identity(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(Identity, self).__init__(menu_container, config, parent, id, nn.Identity, IdentityMenu, pos,
                                       name='Identity', color=QColor(196, 196, 196))


class LinearMenu(LayerMenu):
    def get_parameters_widget(self):
        layout = QGridLayout()

        self.in_features = QSpinBox()
        self.in_features.setMaximum(999999)
        self.in_features.setMinimum(1)
        layout.addWidget(QLabel('in_features:'), 0, 0)
        layout.addWidget(self.in_features, 0, 1)

        self.out_features = QSpinBox()
        self.out_features.setMaximum(999999)
        self.out_features.setMinimum(1)
        layout.addWidget(QLabel('out_features:'), 1, 0)
        layout.addWidget(self.out_features, 1, 1)

        bias = QCheckBox()
        bias.setChecked(True)
        layout.addWidget(QLabel('bias:'), 2, 0)
        layout.addWidget(bias, 2, 1)

        widget = QWidget()
        widget.setLayout(layout)
        return widget


class Linear(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(Linear, self).__init__(menu_container, config, parent, id, nn.Linear, LinearMenu, pos,
                                     name='Linear', color=QColor(96, 255, 96))

