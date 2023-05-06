from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.movable_widget import MovableWidget

from typing import Type
from loguru import logger
from torch.nn import Module


class LayerMenu(MenuWidget):
    def __init__(self, config: Config, name: str):
        super(LayerMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        label = QLabel(f'{name}\n{self.description()}\nПараметры:')
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setMaximumHeight(120)
        self.params = {}
        self.parameters()
        self.layout().addWidget(label)
        self.layout().addWidget(self.build_parameters_widget())

    @property
    def description(self):
        return 'Нет описания'

    def parameters(self):
        pass

    def build_parameters_widget(self):
        layout = QGridLayout()
        for n, (name, widget) in enumerate(self.params.items()):
            layout.addWidget(QLabel(name), n, 0)
            layout.addWidget(widget, n, 1)
        widget = QWidget()
        widget.setLayout(layout)
        return widget


class Layer(MovableWidget):
    chosen = pyqtSignal()
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 module: Type[Module] = Module,
                 menu: Type[LayerMenu] = LayerMenu,
                 pos: QPoint = QPoint(10, 10),
                 name: str = 'EmptyLayer',
                 color: QColor = QColor(210, 210, 210)):
        self.config = config
        self.menu = menu(config, f'{name}_{id}')
        self.menu_container = menu_container
        self.color = color
        self.module = module

        self.name = QLabel(f'{name}_{id}')
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = QFont()
        font.setPointSize(8)

        self.in_button = QPushButton('in')
        self.in_button.setFixedSize(24, 24)
        self.in_button.setFont(font)

        self.out_button = QPushButton('out')
        self.out_button.setFixedSize(24, 24)
        self.out_button.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.in_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name)
        layout.addWidget(self.out_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        widget = QWidget()
        widget.setLayout(layout)

        super(Layer, self).__init__(parent, pos, widget)
        self.setMinimumSize(120, 70)
        self.childWidget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def __str__(self):
        return self.name.text()

    def activate_menu(self):
        logger.debug(f'activate menu {self.menu}')
        self.menu_container.set_menu(self.menu)

    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(e.rect(), self.color)
        rect = e.rect()
        rect.adjust(0, 0, -1, -1)
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect)

    def focusInEvent(self, a0: QFocusEvent):
        super(Layer, self).focusInEvent(a0)
        self.activate_menu()
        self.chosen.emit()


class InputLayerMenu(LayerMenu):
    def description(self):
        return 'Псевдослой, отсюда вылезают данные'


class OutputLayerMenu(LayerMenu):
    def description(self):
        return 'Псевдослой, сюда влезают данные'


class InputLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(InputLayer, self).__init__(menu_container, config, parent, id, Module, InputLayerMenu, pos,
                                     name='Input layer')
        self.in_button.setDisabled(True)


class OutputLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(OutputLayer, self).__init__(menu_container, config, parent, id, Module, OutputLayerMenu, pos,
                                     name='Output layer')
        self.out_button.setDisabled(True)