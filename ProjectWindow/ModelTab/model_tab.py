from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, WidgetWithTabs, MenuContainer
from ProjectWindow.ModelTab.ConstructorWidget.constructor_widget import ConstructorWidget


class ModelTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ModelTab, self).__init__(menu_container, config)

        self.constructor = ConstructorWidget(menu_container, config)

        self.addTab(self.constructor, 'Конструктор')