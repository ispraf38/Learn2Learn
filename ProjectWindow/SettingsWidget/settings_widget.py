from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuWidget, MenuContainer, Config


class SettingsMenu(MenuWidget):
    def __init__(self, config: Config):
        super(SettingsMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel('Меню настроек.\n'
                                       'Здесь можно менять параметры, если выбрать что-нибудь слева.\n'
                                       '(Не работает)'))


class SettingsWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(SettingsWidget, self).__init__(menu_container, config, SettingsMenu)
        self.layout_ = QGridLayout()
        self.setLayout(self.layout_)
        for i, (k, v) in enumerate(config.__dict__.items()):
            self.layout_.addWidget(QLabel(k), i, 0)
            self.layout_.addWidget(QLabel(v), i, 1)