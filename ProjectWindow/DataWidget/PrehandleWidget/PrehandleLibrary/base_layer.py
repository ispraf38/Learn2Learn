from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from utils import RangeSpinbox, FixedMultiSpinbox, MultiSpinBox

from typing import Type, Any
import albumentations as A


class PrehandleLayerMenu(MenuWidget):
    def __init__(self, config: Config):
        super(PrehandleLayerMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        label = QLabel(f'{self.name}\n{self.description}\nПараметры:')
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setMaximumHeight(120)
        self.params = {}
        self.parameters()
        self.layout().addWidget(label)
        self.layout().addWidget(self.build_parameters_widget())

    @property
    def name(self):
        return 'EmptyPrehandle'

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


class PrehandleLayer(WidgetWithMenu):
    def __init__(self,
                 menu_container: MenuContainer,
                 config: Config,
                 menu: Type[PrehandleLayerMenu] = PrehandleLayerMenu,
                 name: str = 'EmptyPrehandleLayer',
                 transform: Any = lambda x: x):
        super(PrehandleLayer, self).__init__(menu_container, config, menu)
        self.button = QPushButton(name)
        self.button.clicked.connect(self.activate_menu)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.button)

        self.transform = transform

    def get_params(self):
        params = {}
        for k, v in self.menu.params.items():
            if isinstance(v, (QDoubleSpinBox, QSpinBox)):
                params[k] = v.value()
            elif isinstance(v, QCheckBox):
                params[k] = v.isChecked()
            elif isinstance(v, (RangeSpinbox, FixedMultiSpinbox, MultiSpinBox)):
                params[k] = v.get_value()
            elif isinstance(v, QComboBox):
                params[k] = v.currentText()
            else:
                raise TypeError(f'Unknown type: {type(v)}')
        return params