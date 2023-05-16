from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import json
import os

from ProjectWindow.utils import Config, MenuContainer, WidgetWithTabs
from ProjectWindow.SettingsWidget.settings_widget import SettingsWidget
from ProjectWindow.DataTab.data_tab import DataTab
from ProjectWindow.ModelTab.model_tab import ModelTab

from loguru import logger


class MainWidget(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(MainWidget, self).__init__(menu_container, config)
        logger.info('Initializing MainWidget')

        self.settings = SettingsWidget(self.menu_container, self.config)

        self.data = DataTab(self.menu_container, self.config)

        self.model = ModelTab(self.menu_container, self.config)

        self.setTabPosition(QTabWidget.TabPosition.West)
        self.addTab(self.settings, 'Настройки')
        self.addTab(self.data, 'Данные')
        self.addTab(self.model, 'Модель')

        self.currentChanged.connect(self.data.gallery.update_visible_images)
        self.data.prehandle.save_button.clicked.connect(self.set_input_layer_dataloader)

    def set_input_layer_dataloader(self):
        self.model.constructor.input_layer.state.not_checked()
        self.data.reset_dataloader()
        self.model.constructor.input_layer.set_dataloader(self.data.train_dataloader, self.data.val_dataloader)


class ProjectWindow(QMainWindow):
    def __init__(self, main_config: Config, project_config: Config):
        super(ProjectWindow, self).__init__()
        self.main_config = main_config
        self.config = project_config
        self.config.data_path = self.main_config.project_data_default_path.replace('<project_name>',
                                                                                   self.config.project_name)

        self.init_config()

        self.setWindowTitle(f'2L project: {self.config.project_name}')

        self.menu_container = MenuContainer()
        self.main_widget = MainWidget(self.menu_container, self.config)

        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.main_widget)
        widget.layout().addWidget(self.menu_container)
        self.setCentralWidget(widget)

        self.showMaximized()

    def init_config(self):
        if not hasattr(self.config, 'data_path'):
            self.config.data_path = self.main_config.project_data_default_path.replace('<project_name>',
                                                                                       self.config.project_name)
            if not os.path.exists(self.config.data_path):
                os.mkdir(self.config.data_path)

        if not hasattr(self.config, 'models_path'):
            self.config.models_path = self.main_config.project_models_default_path.replace('<project_name>',
                                                                                       self.config.project_name)
            if not os.path.exists(self.config.models_path):
                os.mkdir(self.config.models_path)

        if not hasattr(self.config, 'json_path'):
            self.config.json_path = os.path.join(self.config.data_path, self.main_config.data_json_default_name)

        if not hasattr(self.config, 'current_model_file'):
            self.config.current_model_file = os.path.join(self.config.models_path, self.main_config.current_model_file)

        for atr in ['parameter_widgets', 'batch_size', 'val_frac']:
            if not hasattr(self.config, atr):
                self.config.__setattr__(atr, self.main_config.__getattribute__(atr))



