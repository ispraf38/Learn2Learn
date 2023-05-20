from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.movable_widget import MovableWidget
from utils import get_params_from_widget, get_signal_from_widget, set_widget_param

from typing import Type, List, Optional, Dict, Tuple, Any
from loguru import logger
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


class LayerState:
    STATES = {
        'not_checked': {
            'color': QColor(0, 0, 0)
        },
        'ok': {
            'color': QColor(0, 192, 0)
        },
        'error': {
            'color': QColor(192, 0, 0)
        },
        'no_input': {
            'color': QColor(255, 255, 0)
        },
        'compile_error': {
            'color': QColor(255, 0, 0)
        },
        'compiled': {
            'color': QColor(64, 255, 64)
        }
    }

    def __init__(self):
        self.state = 'not_checked'

    def not_checked(self):
        self.state = 'not_checked'

    def ok(self):
        self.state = 'ok'

    def error(self):
        self.state = 'error'

    def no_input(self):
        self.state = 'no_input'

    def compile_error(self):
        self.state = 'compile_error'

    def compiled(self):
        self.state = 'compiled'

    @property
    def color(self):
        return self.STATES[self.state]['color']


class LayerMenu(MenuWidget):
    param_changed = pyqtSignal()

    def __init__(self, config: Config, name: str):
        super(LayerMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        label = QLabel(f'{name}\n{self.description}\nПараметры:')
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setMaximumHeight(120)
        self.layout().addWidget(label)

        self.params = {}
        self.parameters()
        self.layout().addWidget(self.build_parameters_widget())

        self.in_info = QWidget()
        self.in_info.setLayout(QStackedLayout())
        self.out_info = QWidget()
        self.out_info.setLayout(QStackedLayout())
        self.layout().addWidget(self.in_info)
        self.layout().addWidget(self.out_info)

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
            get_signal_from_widget(widget).connect(self.param_changed.emit)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def set_in_info(self, inputs: Dict[str, Any]):
        if self.in_info.layout().currentWidget() is not None:
            self.in_info.layout().currentWidget().close()
        layout = QGridLayout()
        layout.addWidget(QLabel('Inputs:'), 0, 0)
        for n, (k, v) in enumerate(inputs.items()):
            layout.addWidget(QLabel(k), n + 1, 1)
            if v is None:
                widget = (QLabel('None'))
            elif isinstance(v, torch.Tensor):
                widget = QLabel(f'Тензор размерности {v.shape}')
            else:
                widget = QLabel('Неизвестный тип данных')
            layout.addWidget(widget, n + 1, 2)
        widget = QWidget()
        widget.setLayout(layout)
        self.in_info.layout().addWidget(widget)
        self.in_info.layout().setCurrentWidget(widget)

    def set_out_info(self, outputs: Dict[str, Any]):
        if self.out_info.layout().currentWidget() is not None:
            self.out_info.layout().currentWidget().close()
        layout = QGridLayout()
        layout.addWidget(QLabel('Outputs:'), 0, 0)
        for n, (k, v) in enumerate(outputs.items()):
            layout.addWidget(QLabel(k), n + 1, 1)
            if v is None:
                widget = (QLabel('None'))
            elif isinstance(v, torch.Tensor):
                widget = QLabel(f'Тензор размерности {v.shape}')
            else:
                widget = QLabel('Неизвестный тип данных')
            layout.addWidget(widget, n + 1, 2)
        widget = QWidget()
        widget.setLayout(layout)
        self.out_info.layout().addWidget(widget)
        self.out_info.layout().setCurrentWidget(widget)


class LayerButton(QPushButton):
    click = pyqtSignal(str)

    def __init__(self, text):
        super(LayerButton, self).__init__()
        self.setText(text)
        font = QFont()
        font.setPointSize(8)

        self.setFont(font)
        self.setFixedHeight(20)
        metrics = QFontMetrics(font)
        bounding_rect = metrics.boundingRect(text)
        self.setFixedWidth(bounding_rect.width() + 8)
        self.setStyleSheet('text-align: left top;'
                           'padding-left: 2px;'
                           'padding-top: 2px;')

        self.clicked.connect(lambda: self.click.emit(text))


class Layer(MovableWidget):
    chosen = pyqtSignal()
    in_click = pyqtSignal(str)
    out_click = pyqtSignal(str)

    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 module: Type[Module] = Module,
                 menu: Type[LayerMenu] = LayerMenu,
                 pos: QPoint = QPoint(10, 10),
                 name: str = 'EmptyLayer',
                 color: QColor = QColor(210, 210, 210),
                 in_buttons: Optional[List[str]] = None,
                 out_buttons: Optional[List[str]] = None):
        if in_buttons is None:
            in_buttons = ['in']
        if out_buttons is None:
            out_buttons = ['out']

        self.inputs = in_buttons
        self.outputs = out_buttons

        self.type_name = name
        self.id = id
        self.config = config
        self.menu = menu(config, self.full_name)
        self.menu_container = menu_container
        self.color = color
        self.module = module
        self.F = None
        self.current_output = {i: None for i in self.outputs}
        self.current_input = {i: None for i in self.inputs}
        self.state = LayerState()
        self.previous_layers: Dict[str, Tuple[Optional[Layer], Optional[str]]] = {i: (None, None) for i in self.inputs}
        self.next_layers: Dict[str, List[Tuple[Optional[Layer], Optional[str]]]] = {i: [] for i in self.outputs}

        self.name = QLabel(self.full_name)
        metrics = QFontMetrics(self.name.font())
        bounding_rect = metrics.boundingRect(self.full_name)
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name.setStyleSheet('padding: 0px')
        self.name.setFixedHeight(bounding_rect.height() + 10)

        self.in_buttons = {}
        for i in self.inputs:
            button = LayerButton(i)
            self.in_buttons[i] = button

        self.out_buttons = {}
        for i in self.outputs:
            button = LayerButton(i)
            self.out_buttons[i] = button

        in_buttons_widget = QWidget()
        in_buttons_widget.setLayout(QHBoxLayout())
        in_buttons_widget.layout().setContentsMargins(0, 0, 0, 0)
        for button in self.in_buttons.values():
            in_buttons_widget.layout().addWidget(button, alignment=Qt.AlignmentFlag.AlignTop)

        out_buttons_widget = QWidget()
        out_buttons_widget.setLayout(QHBoxLayout())
        out_buttons_widget.setContentsMargins(0, 0, 0, 0)
        for button in self.out_buttons.values():
            out_buttons_widget.layout().addWidget(button, alignment=Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.addWidget(in_buttons_widget, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.name, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(out_buttons_widget, alignment=Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        widget = QWidget()
        widget.setLayout(layout)

        super(Layer, self).__init__(parent, pos, widget)

        self.setStyleSheet('padding: 0px;'
                           'text-align: center')

        for button in self.in_buttons.values():
            button.click.connect(self.in_click)

        for button in self.out_buttons.values():
            button.click.connect(self.out_click)

        self.setFixedSize(120, 90)
        in_buttons_widget.setFixedHeight((self.height() - self.name.height()) // 2)
        out_buttons_widget.setFixedHeight((self.height() - self.name.height()) // 2)
        self.childWidget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.menu.param_changed.connect(self.set_not_checked)

    @property
    def full_name(self):
        return f'{self.type_name}_{self.id}'

    def set_not_checked(self):
        self.state.not_checked()
        self.update()

    def get_params(self):
        return {k: get_params_from_widget(v) for k, v in self.menu.params.items()}

    def __str__(self):
        return self.name.text()

    def activate_menu(self):
        self.menu_container.set_menu(self.menu)

    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(e.rect(), self.color)
        rect = e.rect()
        rect.adjust(1, 1, -1, -1)
        pen = QPen()
        pen.setColor(self.state.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)

    def focusInEvent(self, a0: QFocusEvent):
        super(Layer, self).focusInEvent(a0)
        self.activate_menu()
        self.chosen.emit()

    def compile(self) -> bool:
        try:
            self.F = self.module(**self.get_params())
        except Exception as e:
            logger.error(e)
            self.state.compile_error()
            self.update()
            logger.error(f'Compilation failed: {self}')
            return False
        else:
            self.state.compiled()
            self.update()
            logger.success(f'Compilation succeeded: {self}')
            return True

    def update_in_out_menu(self):
        self.menu.set_out_info(self.current_output)
        for o, next_layers in self.next_layers.items():
            for l, i in next_layers:
                if l is not None:
                    l.current_input[i] = self.current_output[o]
                    l.menu.set_in_info(l.current_input)

    def forward_test(self, x) -> Tuple[Dict[str, Any], bool]:
        try:
            out = self.forward(x)
        except Exception as e:
            logger.error(e)
            self.state.error()
            self.update()
            logger.error(f'Forward test failed: {self}')
            return {}, False
        else:
            self.state.ok()
            self.update()
            self.current_output = out
            self.update_in_out_menu()
            logger.success(f'Forward test succeeded: {self}')
            return self.current_output, True

    def forward(self, x):
        return {'out': self.F(x['in'])}

    def get_dict(self):
        return self.full_name, {
            'pos': (self.pos().x(), self.pos().y()),
            'params_values': {
                k: v for k, v in self.get_params().items()
            },
            'next_layers': {
                k: [(l.full_name, b) for l, b in v] for k, v in self.next_layers.items()
            },
            'previous_layers': {
                k: (v[0].full_name if v[0] is not None else None, v[1]) for k, v in self.previous_layers.items()
            }
        }

    def load_params_values(self, d):
        for k, v in d.items():
            set_widget_param(self.menu.params[k], v)


class InputLayerMenu(LayerMenu):
    @property
    def description(self):
        return 'Псевдослой, отсюда вылезают данные'


class OutputLayerMenu(LayerMenu):
    @property
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
                                         name='InputLayer', in_buttons=[])
        self.train_dataloader = None
        self.val_dataloader = None
        self.current_label = None

    def set_dataloader(self, train: DataLoader, val: DataLoader):
        self.train_dataloader = train
        self.val_dataloader = val

    def compile(self):
        try:
            for batch in self.train_dataloader:
                # pass
                break
            for batch in self.val_dataloader:
                # pass
                break
        except Exception as e:
            logger.error(e)
            self.state.compile_error()
            self.update()
            logger.error(f'Compilation failed: {self}')
            return False
        else:
            self.state.compiled()
            self.update()
            logger.success(f'Compilation succeeded: {self}')
            return True

    def forward_test(self, x) -> Tuple[Dict[str, Any], bool]:
        out = None
        try:
            for o in self.train_dataloader:
                out = o
                break
        except Exception as e:
            logger.error(e)
            self.state.error()
            self.update()
            logger.error(f'Forward test failed: {self}')
            return {}, False
        else:
            self.state.ok()
            self.update()
            self.current_output = {'out': out[0]}
            self.current_label = out[1]
            self.update_in_out_menu()
            logger.success(f'Forward test succeeded: {self}')
            return self.current_output, True


class OutputLayer(Layer):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 parent: QWidget,
                 id: int,
                 pos: QPoint = QPoint(10, 10)):
        super(OutputLayer, self).__init__(menu_container, config, parent, id, Module, OutputLayerMenu, pos,
                                          name='OutputLayer', out_buttons=[])
        self.loss = None
        self.loss_widget = None

    def set_loss_optim(self, loss):
        self.loss_widget = loss

    def compile(self) -> bool:
        try:
            self.loss = self.loss_widget.parameter.compile()
        except Exception as e:
            logger.error(e)
            self.state.compile_error()
            self.update()
            logger.error(f'Compilation failed: {self}')
            return False
        else:
            self.state.compiled()
            self.update()
            logger.success(f'Compilation succeeded: {self}')
            return True

    def forward_test(self, x, label) -> Tuple[Dict[str, Any], bool]:
        try:
            loss = self.loss(x['in'], label)
            loss.backward()
        except Exception as e:
            logger.error(e)
            self.state.error()
            self.update()
            logger.error(f'Forward test failed: {self}')
            return {}, False
        else:
            self.state.ok()
            self.update()
            self.current_output = {}
            self.update_in_out_menu()
            logger.success(f'Forward test succeeded: {self}')
            return self.current_output, True
