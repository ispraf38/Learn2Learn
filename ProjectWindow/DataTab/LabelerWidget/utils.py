from loguru import logger

from ProjectWindow.movable_widget import MovableWidget

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from typing import Tuple, Union, Dict


class Label(MovableWidget):
    is_editing = pyqtSignal()

    COLORS = {
        'fixed': (255, 255, 255, 16),
        'ready': (255, 255, 255, 16),
        'editing': (0, 128, 255, 16),
        'just_created': (0, 128, 255, 16)
    }

    def __init__(self, parent: QWidget,
                 pos: Union[Tuple[QPoint, QPoint], Dict[str, int]],
                 cWidget: QWidget,
                 state: str = 'just_created'):
        assert state in self.COLORS
        if type(pos) is not dict:
            pos = {
                'l': min(pos[0].x(), pos[1].x()),
                'r': max(pos[0].x(), pos[1].x()),
                't': min(pos[0].y(), pos[1].y()),
                'b': max(pos[0].y(), pos[1].y())
            }
        self.state = state

        super().__init__(parent, QPoint(pos['l'], pos['t']), cWidget)

        self.resize(pos['r'] - pos['l'], pos['b'] - pos['t'])
        self.move(pos['l'], pos['t'])
        self.coordinates = pos
        self.m_isEditing = True
        self.newGeometry.connect(lambda x: self.update_coordinates(x))

    def update_coordinates(self, rect):
        self.coordinates = {
            'l': self.geometry().x(),
            'r': self.geometry().x() + self.width(),
            't': self.geometry().y(),
            'b': self.geometry().y() + self.height()
        }

    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.SolidLine)
        painter.fillRect(e.rect(), QColor(*self.COLORS[self.state]))

        rect = e.rect()
        rect.adjust(1, 1, -1, -1)
        painter.setPen(QColor(*self.COLORS[self.state][:-1]))
        painter.drawRect(rect)

    def focusOutEvent(self, a0: QFocusEvent):
        super().focusOutEvent(a0=a0)
        logger.debug(f'Focus out {self}')
        if self.state == 'editing':
            self.setReadyState()
        elif self.state == 'just_created':
            self.setFixedState()

    def setFixedState(self):
        logger.debug(f'{self} state: fixed')
        self.state = 'fixed'
        self.m_isEditing = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update()

    def setReadyState(self):
        logger.debug(f'{self} state: ready')
        self.state = 'ready'
        self.m_isEditing = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update()

    def setEditingState(self):
        if self.state != 'editing':
            logger.debug(f'{self} state: editing')
        self.state = 'editing'
        self.m_isEditing = True
        self.is_editing.emit()
        self.update()

    def mousePressEvent(self, e: QMouseEvent):
        super().mousePressEvent(e)
        logger.debug('Clicked')
        if self.state == 'ready':
            self.setEditingState()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.button = QPushButton('Click me')
        self.button.setCheckable(True)
        self.button.setChecked(False)

        label = QLabel()
        label.setFixedSize(800, 800)

        self.label = Label(parent=label, pos=(QPoint(10), QPoint(10)), cWidget=QWidget())

        self.button.clicked.connect(lambda x: self.label.__setattr__('m_isEditing', x))

        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.button)
        widget.layout().addWidget(label)
        self.setCentralWidget(widget)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
