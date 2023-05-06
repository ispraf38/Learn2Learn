from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import os
import json
from loguru import logger

from ProjectWindow.utils import Config
from ProjectWindow.project_window import ProjectWindow

MAIN_CONFIG_NAME = 'main_config.json'


class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config(MAIN_CONFIG_NAME)
        self.projects_running = []
        self.setWindowTitle('Learn2Learn')

        label1 = QLabel(f'Папка с проектами:')
        label2 = QLabel(self.config.projects_path)

        self.button_new = QPushButton('Создать новый проект')
        self.new_project_name = QLineEdit()

        self.button_new.clicked.connect(self.create_project)

        self.button_existing = QPushButton('Открыть проект')
        self.button_existing.clicked.connect(self.open_project)
        self.existing_names_list = QListWidget()
        self.get_existing_projects()

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(label2, 0, 1)
        layout.addWidget(self.button_new, 1, 0)
        layout.addWidget(self.new_project_name, 1, 1)
        layout.addWidget(self.button_existing, 2, 0)
        layout.addWidget(self.existing_names_list, 2, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def create_project(self):
        new_name = self.new_project_name.text()
        if new_name == '':
            logger.error('Empty project name')
            QToolTip.showText(QPoint(self.pos().x() + self.new_project_name.pos().x(),
                                     self.pos().y() + self.new_project_name.pos().y()),
                              'Имя проекта не может быть пустым')
            return
        elif new_name in [self.existing_names_list.item(x).text() for x in range(self.existing_names_list.count())]:
            logger.error('Project already exists')
            QToolTip.showText(QPoint(self.pos().x() + self.new_project_name.pos().x(),
                                     self.pos().y() + self.new_project_name.pos().y()),
                              'Проект с таким именем уже существует')
            return
        else:
            new_project_path = os.path.join(self.config.projects_path, self.new_project_name.text())
            if not os.path.exists(new_project_path):
                os.mkdir(new_project_path)
            project_config = Config(os.path.join(new_project_path, self.config.project_config_name))
            project_config.project_path = new_project_path
            project_config.project_name = new_name
            project_config.prehandle_default_test_image = self.config.prehandle_default_test_image

            new_project = ProjectWindow(self.config, project_config)
            self.projects_running.append(new_project)
            self.get_existing_projects()

    def get_existing_projects(self):
        self.existing_names_list.clear()
        for dir in os.listdir(self.config.projects_path):
            if os.path.exists(os.path.join(self.config.projects_path, dir, self.config.project_config_name)):
                self.existing_names_list.addItem(dir)

    def open_project(self):
        chosen_project = self.existing_names_list.currentItem()
        logger.debug(chosen_project)
        if chosen_project is None:
            logger.error('Choose project from list')
            QToolTip.showText(QPoint(self.pos().x() + self.existing_names_list.pos().x(),
                                     self.pos().y() + self.existing_names_list.pos().y()),
                              'Выберете проект из списка')
            return
        else:
            project_config = Config(os.path.join(self.config.projects_path,
                                                 chosen_project.text(),
                                                 self.config.project_config_name))
            project = ProjectWindow(self.config, project_config)
            self.projects_running.append(project)


if __name__ == '__main__':
    app = QApplication([])

    window = StartWindow()
    window.setGeometry(300, 300, 600, 400)
    window.show()

    app.exec()