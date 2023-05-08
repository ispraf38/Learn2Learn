from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import MultiSpinBox

import torch.nn as nn


class DropoutMenu(LayerMenu):
    def description(self):
        return 'Во время обучения зануляет случайные элементы входа'

    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        self.params = {
            'p': p
        }


class DropoutLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(DropoutLayer, self).__init__(menu_container, config, parent, id, nn.Dropout, DropoutMenu, pos,
                                            name='Dropout', color=QColor(255, 32, 32))