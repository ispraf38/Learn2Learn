from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, WidgetWithTabs, MenuContainer
from ProjectWindow.LearningTab.RunWidget.run_widget import RunWidget


class LearningTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(LearningTab, self).__init__(menu_container, config)

        self.run = RunWidget(menu_container, config)

        self.addTab(self.run, 'Запуск')
