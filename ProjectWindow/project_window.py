from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import json
import os

from ProjectWindow.utils import Config, MenuContainer, WidgetWithTabs
from ProjectWindow.SettingsWidget.settings_widget import SettingsWidget
from ProjectWindow.DataWidget.data_widget import DataWidget
from ProjectWindow.ModelWidget.model_widget import ModelWidget

from loguru import logger


class MainWidget(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(MainWidget, self).__init__(menu_container, config)
        logger.info('Initializing MainWidget')

        self.settings = SettingsWidget(self.menu_container, self.config)

        self.data = DataWidget(self.menu_container, self.config)

        self.model = ModelWidget(self.menu_container, self.config)

        self.setTabPosition(QTabWidget.TabPosition.West)
        self.addTab(self.settings, 'Настройки')
        self.addTab(self.data, 'Данные')
        self.addTab(self.model, 'Модель')

        self.currentChanged.connect(self.data.gallery.update_visible_images)


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

        if not hasattr(self.config, 'json_path'):
            self.config.json_path = os.path.join(self.config.data_path, self.main_config.data_json_default_name)




