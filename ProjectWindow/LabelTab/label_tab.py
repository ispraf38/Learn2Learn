from PyQt6.QtWidgets import QTabWidget

from ProjectWindow.utils import WidgetWithTabs, MenuContainer, Config
from ProjectWindow.LabelTab.GalleryWidget.gallery_widget import GalleryWidget
from ProjectWindow.LabelTab.LabelerWidget.labeler_widget import LabelerWidget
from ProjectWindow.LabelTab.data_handler import JsonHandler

from loguru import logger


class LabelTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(LabelTab, self).__init__(menu_container, config)
        logger.info('Initializing DataTab')
        self.setTabPosition(QTabWidget.TabPosition.West)

        self.json_handler = JsonHandler(config)

        self.gallery = GalleryWidget(self.menu_container, self.config, self.json_handler)
        self.labeler = LabelerWidget(self.menu_container, self.config, self.json_handler)

        self.json_handler.image_changed.connect(self.update_current_image)

        self.addTab(self.gallery, 'Галерея')
        self.addTab(self.labeler, 'Разметка')
        self.currentChanged.connect(self.gallery.update_visible_images)

        self.train_dataloader = None
        self.val_dataloader = None

    def update_current_image(self):
        self.gallery.menu.update_image()
        self.labeler.set_image()