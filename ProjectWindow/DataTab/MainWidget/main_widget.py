from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget
from ProjectWindow.DataTab.MainWidget.utils import ModelType


class DataMainWidgetMenu(MenuWidget):
    def __init__(self, config: Config):
        super(DataMainWidgetMenu, self).__init__(config)


class DataMainWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(DataMainWidget, self).__init__(menu_container, config, DataMainWidgetMenu)
        self.setLayout(QGridLayout())

        model_type_text = QLabel('Текущий тип модели:')
        self.model_type = ModelType(menu_container, config)

        self.layout().addWidget(model_type_text, 0, 0)
        self.layout().addWidget(self.model_type, 0, 1)

        batch_size_text = QLabel('batch_size:')
        self.batch_size = QSpinBox()
        self.batch_size.setMaximum(10000)
        self.batch_size.setMinimum(1)
        self.batch_size.setValue(32)

        self.layout().addWidget(batch_size_text, 1, 0)
        self.layout().addWidget(self.batch_size, 1, 1)

        val_frac_text = QLabel('Процент валидационной выборки')
        self.val_frac = QDoubleSpinBox()
        self.val_frac.setSingleStep(0.05)
        self.val_frac.setMaximum(1)
        self.val_frac.setDecimals(2)
        self.val_frac.setValue(0.2)

        self.layout().addWidget(val_frac_text, 2, 0)
        self.layout().addWidget(self.val_frac, 2, 1)
