import torch
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, LayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import MultiSpinBox

import torch.nn as nn
from loguru import logger


class FlattenMenu(LayerMenu):
    @property
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


class nnSumLayer(nn.Module):
    def forward(self, x, y):
        return x + y


class SumLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Сумма двух тензоров'


class SumLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(SumLayer, self).__init__(menu_container, config, parent, id, nnSumLayer, SumLayerMenu, pos,
                                          name='Sum', color=QColor(196, 196, 196), in_buttons=['x', 'y'])

    def forward(self, x):
        return self.F(**x)


class nnProdLayer(nn.Module):
    def forward(self, x, y):
        return x * y


class ProdLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Произведение двух тензоров'


class ProdLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(ProdLayer, self).__init__(menu_container, config, parent, id, nnProdLayer, ProdLayerMenu, pos,
                                          name='Prod', color=QColor(196, 196, 196), in_buttons=['x', 'y'])

    def forward(self, x):
        return self.F(**x)


class nnUnsqueezeLayer(nn.Module):
    def __init__(self, dim):
        super(nnUnsqueezeLayer, self).__init__()
        self.dim_ = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim_)


class UnsqueezeLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Добавляет размерность тензору'

    def parameters(self):
        dim = QSpinBox()

        self.params = {
            'dim': dim
        }


class UnsqueezeLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(UnsqueezeLayer, self).__init__(menu_container, config, parent, id, nnUnsqueezeLayer, UnsqueezeLayerMenu,
                                             pos, name='Unsqueeze', color=QColor(196, 196, 196))


class nnCatLayer(nn.Module):
    def __init__(self, dim):
        super(nnCatLayer, self).__init__()
        self.dim_ = dim

    def forward(self, x, y):
        return torch.cat((x, y), dim=self.dim_)


class CatLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Конкатенация двух тензоров'

    def parameters(self):
        dim = QSpinBox()

        self.params = {
            'dim': dim
        }


class CatLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(CatLayer, self).__init__(menu_container, config, parent, id, nnCatLayer, CatLayerMenu, pos,
                                          name='Cat', color=QColor(196, 196, 196), in_buttons=['x', 'y'])

    def forward_test(self, x):
        try:
            out = self.F(**x)
        except Exception as e:
            logger.error(e)
            self.state.error()
            self.update()
            logger.error(f'Forward test failed: {self}')
            return {}, False
        else:
            self.state.ok()
            self.update()
            self.current_output = {'out': out}
            self.update_in_out_menu()
            logger.success(f'Forward test succeeded: {self}')
            return self.current_output, True


class nnRepeatLayer(nn.Module):
    def __init__(self, sizes):
        super(nnRepeatLayer, self).__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.repeat(*self.sizes)


class RepeatLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Повторяет тензор вдоль указанных осей'

    def parameters(self):
        sizes = MultiSpinBox()

        self.params = {
            'sizes': sizes
        }


class RepeatLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(RepeatLayer, self).__init__(menu_container, config, parent, id, nnRepeatLayer, RepeatLayerMenu, pos,
                                          name='Repeat', color=QColor(196, 196, 196))
