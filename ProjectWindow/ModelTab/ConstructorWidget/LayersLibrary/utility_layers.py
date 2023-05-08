from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import MultiSpinBox

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import torch.nn as nn


class FlattenMenu(LayerMenu):
    def description(self):
        return 'Снижает размерность входящего тензора'

    def parameters(self):
        start_dim = QSpinBox()
        start_dim.setMinimum(-100)

        end_dim = QSpinBox()
        end_dim.setMinimum(-100)
        end_dim.setValue(-1)

        self.params = {
            'start_dim': start_dim,
            'end_dim': end_dim
        }



class FlattenLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(FlattenLayer, self).__init__(menu_container, config, parent, id, nn.Flatten, FlattenMenu, pos,
                                            name='Flatten', color=QColor(196, 196, 196))