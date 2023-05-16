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
        for layer in self.layers:
            for o, next_layers in layer.next_layers.items():
                for l, i in next_layers:
                    if l is not None:
                        p1 = QPoint(
                            layer.pos().x() + layer.out_buttons[o].pos().x() + layer.out_buttons[o].width() // 2,
                            layer.pos().y() + layer.height())
                        p2 = QPoint(
                            l.pos().x() + l.in_buttons[i].pos().x() + l.in_buttons[i].width() // 2,
                            l.pos().y() + l.in_buttons[i].pos().y())
                        color = QColor(0, 0, 0) if ((layer, o), (l, i)) != self.current_arrow else QColor(0, 0, 255)
                        pen.setColor(color)
                        painter.setPen(pen)
                        painter.drawLine(p1, p2)
        painter.end()
        self.setPixmap(pixmap)

    def in_button_clicked(self, layer: Layer, button: str):
        existing_arrow = (layer.previous_layers[button], (layer, button))
        if self.new_arrow is None:
            self.current_arrow = existing_arrow
        else:
            if existing_arrow[0][0] is not None:
                self.delete_connection(existing_arrow)
            self.create_connection(self.new_arrow[0], self.new_arrow[1], layer, button)
            self.new_arrow = None
        self.update_arrows()

    def create_connection(self, layer1: Layer, button1: str, layer2: Layer, button2: str):
        arrow = ((layer1, button1), (layer2, button2))
        layer1.next_layers[button1].append(arrow[1])
        layer2.previous_layers[button2] = arrow[0]
        self.current_arrow = arrow

    def out_button_clicked(self, layer: Layer, button: str):
        self.new_arrow = (layer, button)

    def delete_current_connection(self):
        self.delete_connection(self.current_arrow)
        self.current_arrow = None
        self.update_arrows()

    def delete_connection(self, arrow: Tuple[Tuple[Layer, str], Tuple[Layer, str]]):
        arrow[1][0].previous_layers[arrow[1][1]] = (None, None)
        arrow[0][0].next_layers[arrow[0][1]].remove(arrow[1])

    def set_current_layer(self, layer: Layer):
        self.current_layer = layer

    def delete_current_layer(self):
        if self.current_layer is not None:
            for i, (l, o) in self.current_layer.previous_layers.items():
                if l is not None:
                    self.delete_connection(((l, o), (self.current_layer, i)))
            for o, next_layers in self.current_layer.next_layers.items():
                for l, i in next_layers:
                    if l is not None:
                        self.delete_connection(((self.current_layer, o), (l, i)))
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
