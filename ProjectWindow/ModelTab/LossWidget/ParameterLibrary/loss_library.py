from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.ModelTab.LossWidget.ParameterLibrary.base_parameter_widget import BaseParameterMenu,\
    BaseParameterWidget

from torch.nn.modules import loss


class MSELossMenu(BaseParameterMenu):
    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def name(self):
        return 'MSELoss'

    @property
    def description(self):
        return 'Средняя квадратическая функция потерь'

    def parameters(self):
        reduction = QComboBox()
        reduction.addItems(['none', 'mean', 'sum'])
        reduction.setCurrentText('mean')

        self.params = {
            'reduction': reduction
        }


class MSELossWidget(BaseParameterWidget):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, loss.MSELoss, MSELossMenu)


class BCELossMenu(BaseParameterMenu):
    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def name(self):
        return 'BCELoss'

    @property
    def description(self):
        return 'Бинарная кросс энтропия'

    def parameters(self):
        reduction = QComboBox()
        reduction.addItems(['none', 'mean', 'sum'])
        reduction.setCurrentText('mean')

        self.params = {
            'reduction': reduction
        }


class BCELossWidget(BaseParameterWidget):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, loss.BCELoss, BCELossMenu)
