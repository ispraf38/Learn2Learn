from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuWidget, MenuContainer, Config
from utils import get_params_from_widget

from typing import Dict, Any, Tuple, Optional
from functools import partial


FONT = QFont('Arial', 16)


def update_config(key, widget, config):
    config.__setattr__(key, get_params_from_widget(widget))


def create_widget_by_config(key: str, descr: Dict[str, Any], config: Config) -> Tuple[QLabel, QWidget]:
    label = QLabel(key)
    label.setFont(FONT)
    widget = QLabel()
    signal = None
    value = config.__getattribute__(key)
    if 'label_text' in descr:
        label.setText(descr['label_text'])
    if 'widget' not in descr or descr['widget'] == 'Label':
        widget.setText(str(value))
    elif descr['widget'] == 'SpinBox':
        widget = QSpinBox()
        if 'max' in descr:
            widget.setMaximum(descr['max'])
        if 'min' in descr:
            widget.setMinimum(descr['min'])
        signal = widget.valueChanged
        widget.setValue(value)
    elif descr['widget'] == 'DoubleSpinBox':
        widget = QDoubleSpinBox()
        if 'single_step' in descr:
            widget.setSingleStep(descr['single_step'])
        if 'max' in descr:
            widget.setMaximum(descr['max'])
        if 'decimals' in descr:
            widget.setDecimals(descr['decimals'])
        signal = widget.valueChanged
        widget.setValue(value)
    widget.setFont(FONT)
    if signal is not None:
        signal.connect(partial(update_config, key, widget, config))
    return label, widget


class SettingsMenu(MenuWidget):
    def __init__(self, config: Config):
        super(SettingsMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())
        label = QLabel('Меню настроек.\nЗдесь можно менять параметры, если выбрать что-нибудь слева.')
        label.setFont(FONT)
        label.setWordWrap(True)
        self.layout().addWidget(label)


class SettingsWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(SettingsWidget, self).__init__(menu_container, config, SettingsMenu)
        self.layout_ = QGridLayout()
        self.setLayout(self.layout_)
        for i, (k, v) in enumerate(config.parameter_widgets.items()):
            label, widget = create_widget_by_config(k, v, config)
            self.layout_.addWidget(label, i, 0)
            self.layout_.addWidget(widget, i, 1)
            if isinstance(v, str):
                self.layout_.addWidget(QLabel(v), i, 1)
            elif isinstance(v, dict):
                pass
            else:
                raise ValueError(f'Could not handle config. key: {k}, value: {v}')
