from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import WidgetWithTabs, MenuContainer, Config
from ProjectWindow.DataTab.data_handler import JsonHandler
from ProjectWindow.DataTab.MainWidget.main_widget import DataMainWidget
from ProjectWindow.DataTab.GalleryWidget.gallery_widget import GalleryWidget
from ProjectWindow.DataTab.LabelerWidget.labeler_widget import LabelerWidget
from ProjectWindow.DataTab.PrehandleWidget.prehandle_widget import PrehandleWidget

from loguru import logger
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class DataTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(DataTab, self).__init__(menu_container, config)
        logger.info('Initializing DataTab')

        self.json_handler = JsonHandler(config)

        self.main_widget = DataMainWidget(self.menu_container, self.config)
        self.gallery = GalleryWidget(self.menu_container, self.config, self.json_handler)
        self.labeler = LabelerWidget(self.menu_container, self.config, self.json_handler)
        self.prehandle = PrehandleWidget(self.menu_container, self.config)

        self.json_handler.image_changed.connect(self.update_current_image)

        self.addTab(self.main_widget, 'Основное')
        self.addTab(self.gallery, 'Галерея')
        self.addTab(self.labeler, 'Разметка')
        self.addTab(self.prehandle, 'Предобработка')
        self.currentChanged.connect(self.gallery.update_visible_images)

        self.train_dataloader = None
        self.val_dataloader = None

    def update_current_image(self):
        self.gallery.menu.update_image()
        self.labeler.set_image()

    def reset_dataloader(self):
        self.prehandle.gen_transform()
        dataset = self.main_widget.model_type.dataset_type(self.config, transform=self.prehandle.transform)
        train_ind, val_ind = train_test_split(range(len(dataset)),
                                                    test_size=self.main_widget.val_frac.value(),
                                                    shuffle=True)
        self.train_dataloader = DataLoader(Subset(dataset, train_ind), batch_size=self.main_widget.batch_size.value(),
                                           shuffle= True)
        self.val_dataloader = DataLoader(Subset(dataset, val_ind), batch_size=self.main_widget.batch_size.value())
