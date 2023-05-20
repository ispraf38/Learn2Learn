from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import torch
from loguru import logger
import pandas as pd


class NNWorker(QObject):
    new_epoch = pyqtSignal(int)
    loss_updated = pyqtSignal(pd.DataFrame)
    progress_bar_update = pyqtSignal(int)

    def __init__(self, loss, optim, num_epochs, train_loader, test_loader, model):
        super(NNWorker, self).__init__()
        self.loss = loss
        self.optim = optim
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.losses = pd.DataFrame([], columns=['train', 'test'])

    @logger.catch
    def run(self):
        for epoch in range(self.num_epochs):
            self.progress_bar_update.emit(0)
            self.new_epoch.emit(epoch + 1)
            train_sum_loss = 0
            for n, (features, labels) in enumerate(self.train_loader):
                output = self.model(features)
                loss_ = self.loss(output, labels)
                train_sum_loss += loss_
                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()

                if (n + 1) % 100 == 0:
                    print(f'Epochs [{epoch + 1}/{self.num_epochs}],'
                          f' Step[{n + 1}/{len(self.train_loader)}], Losses: {loss_.item():.4f}')
                self.progress_bar_update.emit((n + 1) * 100 // len(self.train_loader))

            self.progress_bar_update.emit(0)

            test_sum_loss = 0
            for n, (features, labels) in enumerate(self.test_loader):
                output = self.model(features)
                loss_ = self.loss(output, labels)
                test_sum_loss += loss_
                if (n + 1) % 100 == 0:
                    print(f'Epochs [{epoch + 1}/{self.num_epochs}],'
                          f' Step[{n + 1}/{len(self.test_loader)}], Losses: {loss_.item():.4f}')
                self.progress_bar_update.emit((n + 1) * 100 // len(self.test_loader))

            loss_to_append = pd.DataFrame([[float(train_sum_loss / len(self.train_loader)),
                                           float(test_sum_loss / len(self.test_loader))]], columns=['train', 'test'])

            logger.debug(loss_to_append)
            self.losses = pd.concat([self.losses, loss_to_append], ignore_index=True)
            self.loss_updated.emit(self.losses)