from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, Config, MenuWidget
from ProjectWindow.DataTab.DataloaderWidget.syntax import PythonHighlighter

import traceback
from torch.utils.data import Dataset

DEFAULT_DATALOADER_TEXT = '''
def my_init(cls, transform, *args, **kwargs):
    from torch.utils.data import Dataset
    Dataset.__init__(cls, *args, **kwargs)
    cls.sample = [('a', 1), ('b', 2), ('c', 3)]
    cls.transform = transform

def my_getitem(cls, item):
    return cls.sample[item]

def my_len(cls):
    return len(cls.sample)


attrs = {'__init__': my_init, '__getitem__': my_getitem, '__len__': my_len}'''


FONT = QFont('Ariel', 16)


class DataloaderMenu(MenuWidget):
    def __init__(self, config: Config):
        super().__init__(config)
        self.setLayout(QVBoxLayout())

        self.load = QPushButton('Загрузить')
        self.load.setFont(FONT)
        self.save = QPushButton('Сохранить')
        self.save.setFont(FONT)

        self.layout().addWidget(self.load)
        self.layout().addWidget(self.save)


class DataloaderWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, DataloaderMenu)

        font = QFont()
        font.setPointSize(16)
        self.text_editor = QTextEdit()
        self.text_editor.setFont(font)
        self.text_editor.setText(DEFAULT_DATALOADER_TEXT)

        self.output_log = QTextEdit()
        self.output_log.setFont(font)
        self.output_log.setMaximumHeight(300)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.text_editor)
        self.layout().addWidget(self.output_log)

        self.menu.load.clicked.connect(self.load_file)
        self.menu.save.clicked.connect(self.run)

        self.highlighter = PythonHighlighter(self.text_editor.document())
        # self.text_editor.textChanged.connect(self.highlighter.rehighlight)
        self.dataset_class = None

    def load_file(self):
        file = QFileDialog.getOpenFileName(self, 'Open file', self.config.project_path, 'Python Files (*.py)')[0]
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            self.text_editor.setText(text)
            self.highlighter.rehighlight()

    def run(self):
        text = self.text_editor.toPlainText()
        try:
            exec(f'global attrs; {text}')
        except Exception as e:
            self.output_log.setText(traceback.format_exc())
        else:
            self.output_log.setText('Success!')

            self.dataset_class = type('DataloaderClass', (Dataset,), attrs)
