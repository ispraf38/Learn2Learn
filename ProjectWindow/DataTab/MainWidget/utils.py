from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget
from ProjectWindow.DataTab.MainWidget.datasets import ImageObjectDetectionDataset

from functools import partial


MODEL_TYPES = {
    'Image object detection': ImageObjectDetectionDataset
}


class ModelTypeMenu(MenuWidget):
    chosen = pyqtSignal(str)

    def __init__(self, config: Config):
        super(ModelTypeMenu, self).__init__(config)
        self.setLayout(QVBoxLayout())

        for t in MODEL_TYPES:
            button = QPushButton(t)
            button.clicked.connect(partial(self.chosen.emit, t))
            self.layout().addWidget(button)


class ModelType(WidgetWithMenu):
    chosen = pyqtSignal()

    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ModelType, self).__init__(menu_container, config, ModelTypeMenu)
        self.setLayout(QVBoxLayout())
        self.label = QLabel('Image object detection')
        change = QPushButton('Изменить')
        change.clicked.connect(self.activate_menu)

        self.layout().addWidget(self.label)
        self.layout().addWidget(change)

        self.menu.chosen.connect(lambda x: self.set_text(x))
        self.dataset_type = ImageObjectDetectionDataset

    def set_text(self, text):
        self.label.setText(text)
        self.dataset_type = MODEL_TYPES[text]
        self.chosen.emit()
