from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from loguru import logger
from datetime import datetime

from ProjectWindow.ModelWidget.ConstructorWidget.LayersLibrary.base_layer import Layer


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

    def update(self, *__args):
        super(Canvas, self).update(*__args)
        for layer in self.layers:
            layer.update()

    def update_arrows(self):
        t1 = datetime.now()
        pixmap = self.pixmap.copy()
        t2 = datetime.now()
        painter = QPainter(pixmap)
        pen = QPen()
        pen.setWidth(3)
        for (layer1, layer2) in self.arrows:
            p1 = QPoint(layer1.pos().x() + layer1.out_button.pos().x() + layer1.out_button.width() // 2,
                        layer1.pos().y() + layer1.out_button.pos().y() + layer1.out_button.height())
            p2 = QPoint(layer2.pos().x() + layer2.in_button.pos().x() + layer2.in_button.width() // 2,
                        layer2.pos().y() + layer2.in_button.pos().y())
            color = QColor(0, 0, 0) if (layer1, layer2) != self.current_arrow else QColor(0, 0, 255)
            pen.setColor(color)
            painter.setPen(pen)
            painter.drawLine(p1, p2)
        painter.end()
        t3 = datetime.now()
        self.setPixmap(pixmap)


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
