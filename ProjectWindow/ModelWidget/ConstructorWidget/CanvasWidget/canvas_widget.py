from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from loguru import logger


class Canvas(QLabel):
    def __init__(self):
        super(Canvas, self).__init__()
        self.position = None
        self.scrolling = False

        self.pixmap = QPixmap(10000, 10000)
        self.pixmap.fill(QColor(224, 224, 224))
        self.painter = QPainter(self.pixmap)
        self.painter.setPen(QColor(200, 200, 200))
        for i in range(100):
            self.painter.drawLine(i ** 2, 0, i ** 2, 10000)
            self.painter.drawLine(0, i ** 2, 10000, i ** 2)
        self.painter.end()

        self.setPixmap(self.pixmap)
        self.layers = []

    def update(self, *__args):
        super(Canvas, self).update(*__args)
        for layer in self.layers:
            layer.update()


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
