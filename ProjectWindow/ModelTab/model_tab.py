from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, WidgetWithTabs, MenuContainer
from ProjectWindow.ModelTab.ConstructorWidget.constructor_widget import ConstructorWidget
from ProjectWindow.ModelTab.LossWidget.loss_widget import MainLossWidget


class ModelTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ModelTab, self).__init__(menu_container, config)

        self.constructor = ConstructorWidget(menu_container, config)
        self.loss = MainLossWidget(menu_container, config)

        self.loss.menu.save.clicked.connect(self.set_output_layer_loss_optim)

        self.addTab(self.constructor, 'Конструктор')
        self.addTab(self.loss, 'Параметры обучения')

    def set_output_layer_loss_optim(self):
        self.constructor.output_layer.set_loss_optim(self.loss.loss)
