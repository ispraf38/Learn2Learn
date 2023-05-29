from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from utils import get_params_from_widget

import torch
import torch.nn as nn
from typing import Any


FONT = QFont('Ariel', 16)


def base_loss_function(*args, **kwargs):
    raise NotImplementedError('Функция потерь не выбрана')


class BaseParameterMenu(MenuWidget):
    def __init__(self, config: Config):
        super().__init__(config)
        self.setLayout(QVBoxLayout())
        label = QLabel(f'{self.name}\n{self.description}\nПараметры:')
        label.setFont(FONT)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setMaximumHeight(120)
        self.layout().addWidget(label)

        self.params = {}
        self.parameters()
        self.layout().addWidget(self.build_parameters_widget())

    @property
    def name(self):
        return 'None'

    @property
    def description(self):
        return 'Для выбора функции потерь нажмите кнопку изменить'

    def parameters(self):
        pass

    def build_parameters_widget(self):
        layout = QGridLayout()
        for n, (name, widget) in enumerate(self.params.items()):
            label = QLabel(name)
            label.setFont(FONT)
            widget.setFont(FONT)
            layout.addWidget(label, n, 0)
            layout.addWidget(widget, n, 1)
        widget = QWidget()
        widget.setLayout(layout)
        return widget


class BaseParameterWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config, function: Any = base_loss_function,
                 menu: BaseParameterMenu = BaseParameterMenu):
        super().__init__(menu_container, config, menu)
        self.function = function
        self.param_function = None
        self.button = QPushButton(self.menu.name)
        self.button.setFont(FONT)
        self.button.clicked.connect(self.activate_menu)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.button)

    def get_params(self):
        return {k: get_params_from_widget(v) for k, v in self.menu.params.items()}

    def compile(self):
        self.param_function = self.function(**self.get_params())
        return self.param_function
