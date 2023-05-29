from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import MultiSpinBox

import torch.nn as nn


class Conv1dMenu(LayerMenu):
    def parameters(self):
        in_channels = QSpinBox()
        in_channels.setMaximum(100000)

        out_channels = QSpinBox()
        out_channels.setMaximum(100000)

        kernel_size = MultiSpinBox()

        stride = MultiSpinBox()
        stride.set_value([1])

        padding = MultiSpinBox()
        padding.set_value([0])

        padding_mode = QComboBox()
        padding_mode.addItems(['zeros', 'reflect', 'replicate', 'circular'])
        padding_mode.setCurrentText('zeros')

        dilation = MultiSpinBox()
        dilation.set_value([1])

        groups = QSpinBox()
        groups.setValue(1)

        bias = QCheckBox()
        bias.setChecked(True)

        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'padding_mode': padding_mode,
            'dilation': dilation,
            'groups': groups,
            'bias': bias
        }

    @property
    def description(self):
        return 'Применяет одномерную свертку ко входным данным'


class Conv1dLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(Conv1dLayer, self).__init__(menu_container, config, parent, id, nn.Conv1d, Conv1dMenu, pos,
                                          name='Conv1d', color=QColor(255, 255, 48))


class Conv2dMenu(LayerMenu):
    def parameters(self):
        in_channels = QSpinBox()
        in_channels.setMaximum(100000)

        out_channels = QSpinBox()
        out_channels.setMaximum(100000)

        kernel_size = MultiSpinBox()

        stride = MultiSpinBox()
        stride.set_value([1])

        padding = MultiSpinBox()
        padding.set_value([0])

        padding_mode = QComboBox()
        padding_mode.addItems(['zeros', 'reflect', 'replicate', 'circular'])
        padding_mode.setCurrentText('zeros')

        dilation = MultiSpinBox()
        dilation.set_value([1])

        groups = QSpinBox()
        groups.setValue(1)

        bias = QCheckBox()
        bias.setChecked(True)

        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'padding_mode': padding_mode,
            'dilation': dilation,
            'groups': groups,
            'bias': bias
        }

    @property
    def description(self):
        return 'Применяет двумерную свертку ко входным данным'


class Conv2dLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(Conv2dLayer, self).__init__(menu_container, config, parent, id, nn.Conv2d, Conv2dMenu, pos,
                                          name='Conv2d', color=QColor(255, 255, 96))
