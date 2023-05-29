from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.ModelTab.LossWidget.ParameterLibrary.loss_library import *
from ProjectWindow.ModelTab.LossWidget.ParameterLibrary.optim_library import *
from ProjectWindow.ModelTab.LossWidget.ParameterLibrary.base_parameter_widget import BaseParameterWidget

from typing import Dict
from functools import partial
from loguru import logger

FONT = QFont('Ariel', 16)

LOSSES = {
    'MSELoss': MSELossWidget,
    'BCELoss': BCELossWidget,
}

OPTIMS = {
    'AdamW': AdamWOptimWidget,
    'SGD': SGDOptimWidget,
}


class LearningParameterMenu(MenuWidget):
    parameter_chosen = pyqtSignal(type(BaseParameterWidget))

    def __init__(self, config: Config):
        super().__init__(config)
        self.setLayout(QVBoxLayout())
        self.buttons = {}
        self.custom_parameter = None

    def create_parameters_menu(self, parameters):
        for name, widget in parameters.items():
            button = QPushButton(name)
            button.setFont(FONT)
            button.clicked.connect(partial(self.parameter_chosen.emit, widget))
            self.layout().addWidget(button)


class LearningParameterWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config, paramters: Dict, name: str,
                 custom_parameter: bool = False):
        super().__init__(menu_container, config, LearningParameterMenu)
        if custom_parameter:
            self.menu.custom_parameter = QPushButton('Загрузить функцию ошибки')
            self.menu.custom_parameter.clicked.connect(self.load_parameter)

        self.menu.create_parameters_menu(paramters)
        self.menu.parameter_chosen.connect(self.reset_parameter)

        self.label = QLabel(name)
        self.label.setFont(FONT)

        self.change_parameter_button = QPushButton('Изменить')
        self.change_parameter_button.clicked.connect(self.activate_menu)
        self.change_parameter_button.setFont(FONT)

        self.parameter = BaseParameterWidget(menu_container, config)
        self.parameter.setFont(FONT)
        self.update_layout()

    def load_parameter(self):
        file_name = QFileDialog.getOpenFileName()
        logger.debug(file_name)
        with open(file_name[0], 'r', encoding='utf-8') as f:
            data = f.read()

    def update_layout(self):
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.parameter)
        self.layout().addWidget(self.change_parameter_button)

    def reset_parameter(self, param):
        self.parameter.close()
        self.parameter = param(self.menu_container, self.config)
        self.parameter.activate_menu()
        self.update_layout()


class MainLossMenu(MenuWidget):
    def __init__(self, config: Config):
        super(MainLossMenu, self).__init__(config)
        self.container = MenuContainer()
        self.save = QPushButton('Сохранить')
        self.save.setFont(FONT)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.container)
        self.layout().addWidget(self.save)


class MainLossWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(MainLossWidget, self).__init__(menu_container, config, MainLossMenu)
        self.loss = LearningParameterWidget(self.menu.container, config, LOSSES, 'Функция потерь:')
        self.optim = LearningParameterWidget(self.menu.container, config, OPTIMS, 'Алгоритм оптимизации:')

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.loss)
        self.layout().addWidget(self.optim)
