from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import torch.nn as nn
from utils import MultiSpinBox


class MaxPool2dMenu(LayerMenu):
    def parameters(self):
        kernel_size = MultiSpinBox()

        stride = MultiSpinBox()
        stride.set_value([1])

        padding = MultiSpinBox()
        padding.set_value([0])

        dilation = MultiSpinBox()
        dilation.set_value([1])

        return_indices = QCheckBox()

        ceil_mode = QCheckBox()

        self.params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'return_indices': return_indices,
            'ceil_mode': ceil_mode
        }


class MaxPool2dLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(MaxPool2dLayer, self).__init__(menu_container, config, parent, id, nn.MaxPool2d, MaxPool2dMenu, pos,
                                            name='MaxPool2d', color=QColor(255, 128, 64))