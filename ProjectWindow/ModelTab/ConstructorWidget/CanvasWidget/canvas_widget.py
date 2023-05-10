from typing import Tuple

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from loguru import logger
from datetime import datetime

from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer


class Canvas(QLabel):
    def __init__(self):
        super(Canvas, self).__init__()

        self.pixmap = QPixmap(3000, 3000)
        self.pixmap.fill(QColor(224, 224, 224))
        self.painter = QPainter(self.pixmap)
        self.painter.setPen(QColor(200, 200, 200))
        for i in range(100):
            self.painter.drawLine(i ** 2, 0, i ** 2, 10000)
            self.painter.drawLine(0, i ** 2, 10000, i ** 2)
        self.painter.end()

        self.setPixmap(self.pixmap)
        self.layers = []
        self.arrows = []
        self.new_arrow = None
        self.current_arrow = None
        self.current_layer = None

    def update(self, *__args):
        super(Canvas, self).update(*__args)
        for layer in self.layers:
            layer.update()

    def update_arrows(self):
        pixmap = self.pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen()
        pen.setWidth(3)
        for ((l1, b1), (l2, b2)) in self.arrows:
            p1 = QPoint(l1.pos().x() + l1.out_buttons[b1].pos().x() + l1.out_buttons[b1].width() // 2,
                        l1.pos().y() + l1.height())
            p2 = QPoint(l2.pos().x() + l2.in_buttons[b2].pos().x() + l2.in_buttons[b2].width() // 2,
                        l2.pos().y() + l2.in_buttons[b2].pos().y())
            color = QColor(0, 0, 0) if ((l1, b1), (l2, b2)) != self.current_arrow else QColor(0, 0, 255)
            pen.setColor(color)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
        painter.end()
        self.setPixmap(pixmap)

    def in_button_clicked(self, layer: Layer, button: str):
        existing_arrow = None
        for arrow in self.arrows:
            if arrow[1] == (layer, button):
                existing_arrow = arrow
        if self.new_arrow is None:
            self.current_arrow = existing_arrow
        else:
            if existing_arrow is not None:
                self.arrows.remove(existing_arrow)
            current_arrow = (self.new_arrow, (layer, button))
            layer.previous_layers[button] = self.new_arrow
            self.new_arrow[0].next_layers[self.new_arrow[1]] = (layer, button)
            self.arrows.append(current_arrow)
            self.new_arrow = None
            self.current_arrow = current_arrow
        self.update_arrows()

    def out_button_clicked(self, layer: Layer, button: str):
        self.new_arrow = (layer, button)

    def delete_current_connection(self):
        self.delete_connection(self.current_arrow)
        self.current_arrow = None
        self.update_arrows()

    def delete_connection(self, arrow: Tuple[Tuple[Layer, str], Tuple[Layer, str]]):
        if arrow in self.arrows:
            self.arrows.remove(arrow)
            arrow[1][0].previous_layers[arrow[0][1]] = (None, None)
            arrow[0][0].next_layers[arrow[1][1]] = (None, None)
        else:
            logger.error(f'There are no {arrow} connection')

    def set_current_layer(self, layer: Layer):
        self.current_layer = layer

    def delete_current_layer(self):
        if self.current_layer is not None:
            if self.current_arrow is not None and self.current_layer in self.current_arrow:
                self.delete_current_connection()
            arrows_to_delete = []
            for arrow in self.arrows:
                if self.current_layer in (arrow[0][0], arrow[1][0]):
                    arrows_to_delete.append(arrow)
            for arrow in arrows_to_delete:
                self.delete_connection(arrow)
            self.layers.remove(self.current_layer)
            self.current_layer.close()
            self.current_layer = None
            self.update()
            self.update_arrows()


class CanvasWidget(QScrollArea):
    def __init__(self):
        super(CanvasWidget, self).__init__()
        self.position = None
        self.scrolling = False

        self.canvas = Canvas()
        self.setWidget(self.canvas)
        logger.debug(self.horizontalScrollBar().maximum())
        logger.debug(self.horizontalScrollBar().minimum())

    def mousePressEvent(self, e: QMouseEvent) -> None:
        logger.debug(e.position())
        self.position = QPoint(round(e.globalPosition().x() - self.geometry().x()),
                               round(e.globalPosition().y() - self.geometry().y()))
        self.scrolling = True

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.scrolling = False

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.scrolling:
            assert self.position is not None
            pos = QPoint(round(e.globalPosition().x() - self.geometry().x()),
                         round(e.globalPosition().y() - self.geometry().y()))
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (pos.x() - self.position.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (pos.y() - self.position.y()))
            self.position = pos
            self.canvas.update()
