from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithTabs, MenuContainer, Config
from ProjectWindow.DataWidget.data_handler import JsonHandler
from ProjectWindow.DataWidget.GalleryWidget.gallery_widget import GalleryWidget
from ProjectWindow.DataWidget.LabelerWidget.labeler_widget import LabelerWidget

from loguru import logger


class DataWidget(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(DataWidget, self).__init__(menu_container, config)
        logger.info('Initializing DataWidget')

        self.json_handler = JsonHandler(config)

        self.gallery = GalleryWidget(self.menu_container, self.config, self.json_handler)
        self.labeler = LabelerWidget(self.menu_container, self.config, self.json_handler)
        self.json_handler.image_changed.connect(self.update_current_image)

        self.addTab(self.gallery, 'Галерея')
        self.addTab(self.labeler, 'Разметка')
        self.currentChanged.connect(self.gallery.update_visible_images)

    def update_current_image(self):
        self.gallery.menu.update_image()
        self.labeler.set_image()
