from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *


from loguru import logger


class MultiSpinBox(QWidget):
    value_changed = pyqtSignal()

    def __init__(self):
        super(MultiSpinBox, self).__init__()
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

    def __init__(self, num_spinbox: int):
        super(FixedMultiSpinbox, self).__init__()
        layout = QHBoxLayout()
        self.spinboxes = []

        for i in range(num_spinbox):
            sb = QDoubleSpinBox()
            sb.setMaximum(10)
            sb.setDecimals(4)
            sb.setSingleStep(0.1)
            sb.setValue(1)
            sb.valueChanged.connect(self.value_changed.emit)
            self.spinboxes.append(sb)
            layout.addWidget(sb)

    def set_value(self, value):
        for sb, v in zip(self.spinboxes, value):
            sb.setValue(v)

    def get_value(self):
        value = [sb.value() for sb in self.spinboxes]
        return tuple(value)