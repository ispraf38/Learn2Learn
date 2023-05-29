import functools
from typing import List, Dict, Optional

import os
import cv2
import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from loguru import logger

from ProjectWindow.utils import WidgetWithMenu, MenuContainer, MenuWidget, Config
from ProjectWindow.LabelTab.data_handler import JsonHandler
from ProjectWindow.LabelTab.LabelerWidget.utils import Label


FONT = QFont('Ariel', 16)


class LabelWidget(QLabel):
    updated_rect_signal = pyqtSignal(list, int)
    rect_chosen = pyqtSignal(int)
    MIN_SIDE = 10

    def __init__(self, labels: List[Dict[str, int]], **kwargs):
        super().__init__(**kwargs)
        self.labels = []

        self.set_labels(labels)

        self.new_rect = None
        self.current_rect: Optional[Label] = None
        self.begin = QPoint()
        self.end = QPoint()
        self.flag = False
        self.add_rects_mode = True

    def updated(self):

        index = self.labels.index(self.current_rect) if self.current_rect is not None else -1

        self.updated_rect_signal.emit(self.labels, index)

    def set_labels(self, labels: List[Dict[str, int]]):
        for label in self.labels:
            label.close()
        self.labels.clear()

        for label in labels:
            new_label = Label(self, label, QWidget())
            self.labels.append(new_label)
        for new_label in self.labels:
            new_label.newGeometry.connect(self.updated)
            new_label.is_editing.connect(functools.partial(self.set_current_widget, new_label))
            if self.add_rects_mode:
                new_label.setFixedState()
            else:
                new_label.setReadyState()
            logger.info(self.labels.index(new_label))
        self.current_rect = None

    def set_current_widget(self, widget: Optional[Label] = None, id: Optional[int] = None):
        if widget is not None:
            if self.current_rect is not None and self.current_rect != widget:
                if self.add_rects_mode:
                    self.current_rect.setFixedState()
                else:
                    self.current_rect.setReadyState()
            self.current_rect = widget
            self.rect_chosen.emit(self.labels.index(widget))
        elif id is not None:
            if id >= len(self.labels):
                id -= 1
            if self.current_rect is not None and self.current_rect != self.labels[id]:
                if self.add_rects_mode:
                    self.current_rect.setFixedState()
                else:
                    self.current_rect.setReadyState()
            self.current_rect = self.labels[id]
            self.current_rect.setEditingState()

    def delete_current_widget(self):
        if self.current_rect is None:
            return
        id = self.labels.index(self.current_rect)
        self.labels.remove(self.current_rect)
        self.current_rect.close()
        if self.labels:
            if id >= len(self.labels):
                id -= 1
            self.set_current_widget(id=id)
            self.updated_rect_signal.emit(self.labels, id)
        else:
            self.current_rect = None
            self.updated_rect_signal.emit(self.labels, -1)

    def delete_labels(self):
        for label in self.labels:
            label.close()
        self.labels.clear()
        self.current_rect = None
        self.updated()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.buttons() & Qt.MouseButton.LeftButton:
                logger.info(f'Clicked on widget {obj} with state {obj.state}')
                if obj.state == 'fixed':
                    logger.info(f'{obj}, global pos: {event.globalPosition().toPoint()}, '
                                f'local pos {event.pos()}, position with respect to self'
                                f'{self.mapFromGlobal(obj.mapToGlobal(event.pos()))}')
                    pos = self.mapFromGlobal(obj.mapToGlobal(event.pos()))
                    self.flag = True
                    self.new_rect = [pos, pos]
                    logger.debug(f'{self.new_rect}')
                    self.update()
        return super(LabelWidget, self).eventFilter(obj, event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.new_rect is not None:
            self.draw_rect(self.new_rect[0], self.new_rect[1], 'red')

    def draw_rect(self, pos1: float, pos2: float, color: str):
        rect = QRect(pos1, pos2)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(color), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        logger.info('Mouse clicked')
        if self.add_rects_mode:
            self.flag = True
            self.new_rect = [event.pos(), event.pos()]
            logger.debug(f'{self.new_rect}')
            self.update()

    def mouseMoveEvent(self, event):
        if self.add_rects_mode:
            if self.flag:
                self.new_rect[1] = event.pos()
                self.update()

    def mouseReleaseEvent(self, event):
        logger.info('Mouse released')
        if self.add_rects_mode and self.new_rect:
            if self.flag and min(abs(self.new_rect[0].x() - self.new_rect[1].x()),
                                 abs(self.new_rect[0].y() - self.new_rect[1].y())) > self.MIN_SIDE:
                logger.debug(self.new_rect)
                label = Label(self, self.new_rect, QWidget())
                label.newGeometry.connect(self.updated)
                label.is_editing.connect(lambda: self.set_current_widget(label))
                self.labels.append(label)
                self.set_current_widget(label)
                self.new_rect = None
                self.update()
                self.updated_rect_signal.emit(self.labels, len(self.labels) - 1)
                self.flag = False
            else:
                self.new_rect = None
                self.update()
                self.flag = False


def abs_to_rel_coord(coorinates: Dict[str, int], width, height):
    for i in 'lr':
        coorinates[i] /= width
    for i in 'tb':
        coorinates[i] /= height
    return coorinates


def rel_to_abs_coord(coorinates: Dict[str, int], width, height):
    for i in 'lr':
        coorinates[i] *= width
    for i in 'tb':
        coorinates[i] *= height
    return coorinates


class LabelMenu(MenuWidget):
    def __init__(self, config: Config):
        super().__init__(config)
        self.json_handler = None
        self.setLayout(QVBoxLayout())
        self.add_label_button = QPushButton('Режим добавления меток')
        self.add_label_button.setCheckable(True)
        self.add_label_button.setChecked(True)
        self.add_label_button.setFont(FONT)

        self.remove_label_button = QPushButton('Удалить выбранную метку')
        self.remove_label_button.setFont(FONT)

        self.edit_label_button = QPushButton('Режим изменения меток')
        self.edit_label_button.setCheckable(True)
        self.edit_label_button.setChecked(False)
        self.edit_label_button.setFont(FONT)

        self.remove_all_button = QPushButton("Удалить все метки")
        self.remove_all_button.setFont(FONT)

        self.labels_list = QListWidget()

        self.previous_button = QPushButton('Предыдущий')
        self.previous_button.setFont(FONT)
        self.next_button = QPushButton('Следующий')
        self.next_button.setFont(FONT)

        buttons = QWidget()
        buttons.setLayout(QHBoxLayout())
        buttons.layout().addWidget(self.previous_button)
        buttons.layout().addWidget(self.next_button)

        self.layout().addWidget(self.add_label_button)
        self.layout().addWidget(self.edit_label_button)
        self.layout().addWidget(self.remove_label_button)
        self.layout().addWidget(self.remove_all_button)
        self.layout().addWidget(self.labels_list)
        self.layout().addWidget(buttons)

    def set_json_handler(self, json_handler: JsonHandler):
        self.json_handler = json_handler

    def update_list(self, rects, index, width, height):
        if self.json_handler.current_image:
            rel_rects = [abs_to_rel_coord(r.coordinates.copy(), width, height) for r in rects]
            self.labels_list.clear()
            self.labels_list.addItems([', '.join([f'{k}: {round(v, 3)}' for k, v in r.items()]) for r in rel_rects])
            self.json_handler.data[self.json_handler.current_image]["labels"] = rel_rects
            self.json_handler.data[self.json_handler.current_image]["checked"] = bool(rects)
        if index >= 0:
            self.labels_list.setCurrentRow(index)


class ImageWidget(QWidget):
    def __init__(self, config: Config, json_handler: JsonHandler):
        super().__init__()
        self.config = config
        self.json_handler = json_handler

        self.label_widget = LabelWidget(labels=self.json_handler.labels, parent=self)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label_widget)
        self.image_width = None
        self.image_height = None

    def set_image(self):
        screen_size = QApplication.primaryScreen().size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()

        # self.img = cv2.imread(os.path.join(self.config.data_path, self.json_handler.current_image))
        stream = open(os.path.join(self.config.data_path, self.json_handler.current_image), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        if len(self.img.shape) == 3:
            height, width, bytesPerComponent = self.img.shape
            format = QImage.Format.Format_RGB888
            bytesPerLine = 3 * width
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        elif len(self.img.shape) == 2:
            height, width = self.img.shape
            format = QImage.Format.Format_MonoLSB
            bytesPerLine = width
        else:
            raise TypeError
        qimg = QImage(self.img.data, width, height, bytesPerLine, format)

        self.pixmap = QPixmap.fromImage(qimg).scaledToHeight(int(screen_height * 0.8))

        if self.pixmap.width() > int(0.7 * screen_width):
            self.pixmap = QPixmap.fromImage(qimg).scaledToWidth(int(screen_width * 0.7))

        self.label_widget.setPixmap(self.pixmap)
        self.setFixedSize(self.pixmap.size())
        self.image_width, self.image_height = self.pixmap.width(), self.pixmap.height()
        current_labels = [rel_to_abs_coord(r.copy(), self.image_width, self.image_height)
                          for r in self.json_handler.labels]

        self.label_widget.set_labels(current_labels)
        self.label_widget.current_rect = None


class LabelerWidget(WidgetWithMenu):
    def __init__(self, menu_container: MenuContainer, config: Config, json_handler: JsonHandler):
        super().__init__(menu_container, config, LabelMenu)
        self.menu.set_json_handler(json_handler)
        self.setLayout(QHBoxLayout())

        self.json_handler = json_handler

        # self.images_layout = layout

        self.rects = self.json_handler.labels

        self.image_widget = ImageWidget(self.config, self.json_handler)
        if json_handler.current_image:
            self.image_widget.set_image()

        self.image_widget.label_widget.updated_rect_signal \
            .connect(lambda list, index: self.menu.update_list(list, index, self.image_widget.image_width,
                                                               self.image_widget.image_height))
        self.image_widget.label_widget.rect_chosen \
            .connect(lambda i: self.menu.labels_list.setCurrentRow(i))

        self.menu.add_label_button.toggled.connect(lambda x: self.set_mode(x))
        self.menu.edit_label_button.clicked.connect(lambda x: self.menu.add_label_button.setChecked(not x))
        self.menu.remove_all_button.clicked.connect(lambda: self.remove_labels())

        self.menu.previous_button.clicked.connect(self.json_handler.previous)
        self.menu.next_button.clicked.connect(self.json_handler.next)
        self.menu.remove_label_button.clicked.connect(self.image_widget.label_widget.delete_current_widget)

        self.menu.labels_list.itemSelectionChanged.connect(self.set_current_label)

        self.previous = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.previous.activated.connect(self.json_handler.previous)

        self.previous = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.previous.activated.connect(self.json_handler.next)

        # widget = QWidget()
        # widget.setLayout(QHBoxLayout())
        # widget.layout().addWidget(self.image_widget)
        # widget.layout().addWidget(self.menu, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_widget)

    def set_current_label(self):
        if self.image_widget.label_widget.current_rect is not None:
            self.image_widget.label_widget.set_current_widget(id=self.menu.labels_list.currentRow())

    def set_image(self):
        # self.json_handler.current_image = current_image
        self.image_widget.set_image()
        self.image_widget.label_widget.updated_rect_signal.emit(self.image_widget.label_widget.labels, -1)
        # self.tab.setCurrentWidget(self.tab.widget(self.tab.indexOf(self)))

    def set_mode(self, value: bool):  # True if add label mode is active
        self.menu.edit_label_button.setChecked(not value)
        if value:
            for label in self.image_widget.label_widget.labels:
                label.setFixedState()
        else:
            for label in self.image_widget.label_widget.labels:
                label.setReadyState()

        if self.image_widget.label_widget.current_rect is not None:
            self.image_widget.label_widget.current_rect.setEditingState()
        self.image_widget.label_widget.add_rects_mode = value

    def change_image(self):
        # self.change_checkBox(bool(self.json_handler.get_current_labels()))

        if self.json_handler.current_image:
            self.set_image()
            self.json_handler.save()

    def remove_labels(self):
        if self.json_handler.current_image:
            self.image_widget.label_widget.delete_labels()
            # self.change_checkBox(False)
            self.json_handler.clear_labels()
            self.json_handler.save()
    #
    # def change_checkBox(self, flag):
    #
    #     for index in range(len(self.images_layout)):
    #         widget = self.images_layout.itemAt(index).widget()
    #         widget_name = widget.objectName()
    #
    #         if widget_name == self.json_handler.current_image:
    #             checkbox = widget.findChild(QCheckBox, "checkbox")
    #             self.json_handler.dict[self.json_handler.current_image]["checked"] = flag
    #             checkbox.setChecked(flag)


if __name__ == '__main__':
    app = QApplication([])

    window = LabelerWidget()
    window.show()

    app.exec()
