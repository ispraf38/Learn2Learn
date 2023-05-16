from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import MultiSpinBox

import torch.nn as nn
from loguru import logger


class nnToFloatLayer(nn.Module):
    def forward(self, x):
        return x.float()


class nnToIntLayer(nn.Module):
    def forward(self, x):
        return x.int()


TYPES = {
    'float': nnToFloatLayer,
    'int': nnToIntLayer
}


class ToTypeMenu(LayerMenu):
    @property
    def description(self):
        return 'Снижает размерность входящего тензора'

    def parameters(self):
        type_ = QComboBox()
        type_.addItems(TYPES.keys())
        type_.setCurrentText('float')

        self.params = {
            'type': type_
        }


class ToTypeLayer(Layer):

    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(ToTypeLayer, self).__init__(menu_container, config, parent, id, nnToFloatLayer, ToTypeMenu, pos,
                                          name='ToType', color=QColor(196, 196, 196))

    def compile(self) -> bool:
        try:
            self.module = TYPES[self.get_params()['type']]
            self.F = self.module()
        except Exception as e:
            logger.error(e)
            self.state.compile_error()
            self.update()
            logger.error(f'Compilation failed: {self}')
            return False
        else:
            self.state.compiled()
            self.update()
            logger.success(f'Compilation succeeded: {self}')
            return True
