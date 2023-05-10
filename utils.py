from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from loguru import logger
from typing import Tuple
from functools import partial


def get_params_from_widget(widget: QWidget):
    if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
        return widget.value()
    elif isinstance(widget, QCheckBox):
        return widget.isChecked()
    elif isinstance(widget, (RangeSpinbox, FixedMultiSpinbox, MultiSpinBox)):
        return widget.get_value()
    elif isinstance(widget, QComboBox):
        return widget.currentText()
    else:
        raise TypeError(f'Unknown type: {type(widget)}')


def get_signal_from_widget(widget: QWidget) -> pyqtSignal:
    if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
        return widget.valueChanged
    elif isinstance(widget, QCheckBox):
        return widget.stateChanged
    elif isinstance(widget, (RangeSpinbox, FixedMultiSpinbox, MultiSpinBox)):
        return widget.value_changed
    elif isinstance(widget, QComboBox):
        return widget.currentTextChanged
    else:
        raise TypeError(f'Unknown type: {type(widget)}')


class MultiSpinBox(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, double: bool = False):
        super(MultiSpinBox, self).__init__()
        self.double = double
        self.layout = QHBoxLayout()
        self.spinboxes = []

        self.button_add = QPushButton('+')
        self.button_add.setFixedSize(22, 22)
        self.button_remove = QPushButton('-')
        self.button_remove.setFixedSize(22, 22)
        self.button_remove.setDisabled(True)
        self.add_spinbox()

        self.button_add.clicked.connect(self.add_spinbox)
        self.button_remove.clicked.connect(self.remove_spinbox)

        self.setLayout(self.layout)

    def add_spinbox(self, value=None, rebuild=True):
        if self.double:
            sb = QDoubleSpinBox()
        else:
            sb = QSpinBox()
        sb.setMaximum(1000000)
        if value is not None:
            sb.setValue(value)
        sb.valueChanged.connect(self.value_changed.emit)
        self.spinboxes.append(sb)
        if rebuild:
            self.rebuild_layout()
            self.value_changed.emit()
        if len(self.spinboxes) > 1:
            self.button_remove.setEnabled(True)

    def remove_spinbox(self):
        self.spinboxes.pop()
        self.rebuild_layout()
        if len(self.spinboxes) == 1:
            self.button_remove.setDisabled(True)
        self.value_changed.emit()

    def rebuild_layout(self):
        logger.debug('rebuilding')
        for i in reversed(range(0, self.layout.count())):
            logger.debug(f'Deleting {i}')
            self.layout.itemAt(i).widget().setParent(None)
        for sb in self.spinboxes:
            self.layout.addWidget(sb)
        self.layout.addWidget(self.button_add)
        self.layout.addWidget(self.button_remove)
        logger.success('Rebuild succeed')

    def set_value(self, value: list[int]):
        logger.debug(value)
        self.spinboxes = []
        for i in value:
            self.add_spinbox(i, rebuild=False)
        self.rebuild_layout()
        logger.debug('Emit signal')
        self.value_changed.emit()
        logger.success('Done')

    def get_value(self):
        value = []
        for sb in self.spinboxes:
            value.append(sb.value())
        return value


class FixedMultiSpinbox(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, num_spinbox: int, double: bool = False):
        super(FixedMultiSpinbox, self).__init__()
        layout = QHBoxLayout()
        self.spinboxes = []

        for i in range(num_spinbox):
            if double:
                sb = QDoubleSpinBox()
                sb.setMaximum(10)
                sb.setDecimals(4)
                sb.setSingleStep(0.1)
                sb.setValue(1)
            else:
                sb = QSpinBox()
                sb.setMaximum(10000)
            sb.valueChanged.connect(self.value_changed.emit)
            self.spinboxes.append(sb)
            layout.addWidget(sb)

    def set_value(self, value):
        for sb, v in zip(self.spinboxes, value):
            sb.setValue(v)

    def get_value(self):
        value = [sb.value() for sb in self.spinboxes]
        return tuple(value)


class RangeSpinbox(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, min_value: int = 1, max_value: int = 100, double: bool = False):
        super(RangeSpinbox, self).__init__()
        assert min_value <= max_value
        if double:
            self.left_spinbox = QDoubleSpinBox()
            self.left_spinbox.setSingleStep(0.1)
            self.left_spinbox.setDecimals(2)
        else:
            self.left_spinbox = QSpinBox()
        self.left_spinbox.setMinimum(min_value)
        self.left_spinbox.setMaximum(max_value)
        self.left_spinbox.valueChanged.connect(partial(self.check_value, True))

        if double:
            self.right_spinbox = QDoubleSpinBox()
            self.right_spinbox.setSingleStep(0.1)
            self.right_spinbox.setDecimals(2)
        else:
            self.right_spinbox = QSpinBox()
        self.right_spinbox.setMinimum(min_value)
        self.right_spinbox.setMaximum(max_value)
        self.right_spinbox.valueChanged.connect(partial(self.check_value, False))

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.left_spinbox)
        self.layout().addWidget(self.right_spinbox)

    def set_value(self, values: Tuple[float, float]):
        self.left_spinbox.setValue(values[0])
        self.right_spinbox.setValue(values[1])

    def get_value(self):
        return self.left_spinbox.value(), self.right_spinbox.value()

    def check_value(self, left: bool = True):
        if self.left_spinbox.value() > self.right_spinbox.value():
            if left:
                self.left_spinbox.setValue(self.right_spinbox.value())
            else:
                self.right_spinbox.setValue(self.left_spinbox.value())
