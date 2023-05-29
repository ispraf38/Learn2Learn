from PyQt6.QtWidgets import QTabWidget

from ProjectWindow.utils import WidgetWithTabs, MenuContainer, Config
from ProjectWindow.DataTab.MainWidget.main_widget import DataMainWidget
from ProjectWindow.DataTab.PrehandleWidget.prehandle_widget import PrehandleWidget
from ProjectWindow.DataTab.DataloaderWidget.dataloader_widget import DataloaderWidget

from loguru import logger
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class DataTab(WidgetWithTabs):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(DataTab, self).__init__(menu_container, config)
        logger.info('Initializing DataTab')
        self.setTabPosition(QTabWidget.TabPosition.West)

        self.dataloader = DataloaderWidget(self.menu_container, self.config)
        self.prehandle = PrehandleWidget(self.menu_container, self.config)

        self.addTab(self.dataloader, 'Загрузчик данных')
        self.addTab(self.prehandle, 'Предобработка')

        self.train_dataloader = None
        self.val_dataloader = None

    def reset_dataloader(self):
        self.prehandle.gen_transform()
        if self.dataloader.dataset_class is not None:
            dataset = self.dataloader.dataset_class(transform=self.prehandle.transform)
            train_ind, val_ind = train_test_split(range(len(dataset)),
                                                        test_size=self.config.val_frac,
                                                        shuffle=True)
            self.train_dataloader = DataLoader(Subset(dataset, train_ind), batch_size=self.config.batch_size,
                                               shuffle=True)
            self.val_dataloader = DataLoader(Subset(dataset, val_ind), batch_size=self.config.batch_size)
