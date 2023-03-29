from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import json
from loguru import logger
import os

from typing import Type, Union


class Config:
    def __init__(self, path: str):
        self.__dict__['path'] = path
        if os.path.exists(path):
            self.load(path)
        else:
            self.save()

    def save(self):
        logger.info(f'Saving config: {self.__dict__}')
        with open(self.path, 'w+', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def load(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f'Loading config {config}')
            assert type(config) == dict
            for k, v in config.items():
                self.__dict__[k] = v

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        self.save()


class MenuWidget(QWidget):
    def __init__(self, config: Config):
        super(MenuWidget, self).__init__()
        self.config = config


class EmptyMenuWidget(MenuWidget):
    def __init__(self, config: Config):
        super(EmptyMenuWidget, self).__init__(config)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel('Здесь могла быть Ваша реклама'))


class MenuContainer(QWidget):
    def __init__(self):
        super(MenuContainer, self).__init__()
        self.layout_ = QStackedLayout()
        self.setLayout(self.layout_)
        self.setMaximumWidth(600)
        self.setMinimumWidth(300)

    def set_menu(self, menu: MenuWidget):
        self.layout_.addWidget(menu)
        self.layout_.setCurrentWidget(menu)


class WidgetWithMenu(QWidget):
    def __init__(self, menu_container: MenuContainer, config: Config, menu: Type[MenuWidget] = EmptyMenuWidget):
        super(WidgetWithMenu, self).__init__()
        self.config = config
        self.menu = menu(config)
        self.menu_container = menu_container

    def activate_menu(self):
        self.menu_container.set_menu(self.menu)


class WidgetWithTabs(QTabWidget):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(WidgetWithTabs, self).__init__()
        self.currentChanged.connect(self.activate_menu)
        self.config = config
        self.menu_container = menu_container

    def activate_menu(self):
        if self.count() > 0:
            current_widget = self.widget(self.currentIndex())
            if hasattr(current_widget, 'activate_menu') and callable(getattr(current_widget, 'activate_menu')):
                self.widget(self.currentIndex()).activate_menu()

