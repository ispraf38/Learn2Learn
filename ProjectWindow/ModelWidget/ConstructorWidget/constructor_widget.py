from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer

from ProjectWindow.ModelWidget.ConstructorWidget.CanvasWidget.canvas_widget import CanvasWidget


class ConstructorMenu(MenuWidget):
    def __init__(self, config: Config):
        super(ConstructorMenu, self).__init__(config)


class ConstructorWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ConstructorWidget, self).__init__(menu_container, config, ConstructorMenu)
        self.canvas = CanvasWidget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)
