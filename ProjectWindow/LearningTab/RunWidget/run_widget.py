from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.LearningTab.RunWidget.worker import NNWorker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd


FONT = QFont('Ariel', 16)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class RunMenu(MenuWidget):
    def __init__(self, config: Config):
        super().__init__(config)
        self.run_button = QPushButton('Запуск')
        self.run_button.setFont(FONT)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.run_button)


class RunWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, RunMenu)
        self.worker = None
        self.thread = None
        self.current_epoch = QLabel('Модель не запущена')
        self.current_epoch.setFont(FONT)

        self.loss_plot = MplCanvas(self, width=10, height=8)

        self.toolbar = NavigationToolbar(self.loss_plot)
        self.toolbar.setFont(FONT)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.loss_plot)

        layout = QHBoxLayout()
        layout.addWidget(self.current_epoch)
        layout.addWidget(self.progress_bar)
        widget = QWidget()
        widget.setLayout(layout)
        self.layout().addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

    def update_epoch(self, epoch):
        self.current_epoch.setText(f'Текущая эпоха: {epoch}')

    def update_losses(self, losses: pd.DataFrame):
        self.loss_plot.axes.cla()
        losses.plot(ax=self.loss_plot.axes)
        self.loss_plot.draw()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def run(self, train_dataloader, test_dataloader, loss_widget, optim_widget, model, num_epochs):
        if train_dataloader is None or test_dataloader is None:
            return
        elif loss_widget is None:
            return
        elif optim_widget is None:
            return
        elif model is None:
            return
        loss = loss_widget.compile()
        optim = optim_widget.compile(model.parameters())
        self.worker = NNWorker(loss, optim, num_epochs, train_dataloader, test_dataloader, model)
        self.worker.new_epoch.connect(self.update_epoch)
        self.worker.loss_updated.connect(self.update_losses)
        self.worker.progress_bar_update.connect(self.update_progress_bar)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
