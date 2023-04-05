from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.movable_widget import MovableWidget

from typing import Type
from loguru import logger


class ArrowCircle(QLabel):
    def __init__(self, color):
        super(ArrowCircle, self).__init__()
        self.color = color

    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        painter.setPen(pen)
        painter.drawEllipse(QPoint(10, 10), 10, 10)



class LayerMenu(MenuWidget):
    def __init__(self, config: Config):
        super(LayerMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel('У этого слоя нет параметров'))


class Layer(MovableWidget):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 menu: Type[MenuWidget] = LayerMenu,
                 name: str = 'EmptyLayer',
                 pos: QPoint = QPoint(10, 10),
                 color: QColor = QColor(210, 210, 210)):
        self.config = config
        self.menu = menu(config)
        self.menu_container = menu_container
        self.color = color

        self.name = QLabel(name)
        self.circle = ArrowCircle(color)
        self.circle.setAlignment(Qt.AlignmentFlag.AlignBottom)

        layout = QVBoxLayout()
        layout.addWidget(self.name)
        layout.addWidget(self.circle)
        widget = QWidget()
        widget.setLayout(layout)

        super(Layer, self).__init__(parent, pos, widget)
        self.setMinimumSize(100, 20)

    def activate_menu(self):
        self.menu_container.set_menu(self.menu)

    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(e.rect(), self.color)
        rect = e.rect()
        rect.adjust(1, 1, -1, -1)
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect)

    def focusInEvent(self, a0: QFocusEvent):
        super(Layer, self).focusInEvent(a0)
        self.activate_menu()
