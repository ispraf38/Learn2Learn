"""
created (translated to pyqt) by Aleksandr Korabelnikov (nesoriti@yandex.ru)
origin was written in c++ by Aleksey Osipov (aliks-os@yandex.ru)
wiki: https://wiki.qt.io/Widget-moveable-and-resizeable
distributed without any warranty. Code bellow can contains mistakes taken from c++ version and/or created by my own
"""

import sys
from enum import Enum

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import QPoint, pyqtSignal, QRect
from PyQt6.QtGui import QColor, QCursor, QPainterPath, QBrush
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMenu, QLabel, QMainWindow, QPushButton

from loguru import logger


class Mode(Enum):
    NONE = 0,
    MOVE = 1,
    RESIZETL = 2,
    RESIZET = 3,
    RESIZETR = 4,
    RESIZER = 5,
    RESIZEBR = 6,
    RESIZEB = 7,
    RESIZEBL = 8,
    RESIZEL = 9


class MovableWidget(QWidget):
    """ allow to move and resize by user"""
    # menu = None
    mode = Mode.NONE
    position = None
    inFocus = pyqtSignal(bool)
    outFocus = pyqtSignal(bool)
    newGeometry = pyqtSignal(QRect)

    def __init__(self, parent, p, cWidget):
        super().__init__(parent=parent)

        self.m_infocus = True
        self.m_showMenu = False
        self.m_isEditing = True

        # self.menu = QMenu(parent=self, title='menu')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setVisible(True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.setFocus()
        self.move(p)

        self.vLayout = QVBoxLayout(self)
        self.setChildWidget(cWidget)
        self.installEventFilter(parent)

    def setChildWidget(self, cWidget):
        if cWidget:
            self.childWidget = cWidget
            self.childWidget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.childWidget.setParent(self)
            self.childWidget.releaseMouse()
            self.vLayout.addWidget(cWidget)
            self.vLayout.setContentsMargins(0, 0, 0, 0)

    '''def popupShow(self, pt: QPoint):
        if self.menu.isEmpty:
            return
        global_ = self.mapToGlobal(pt)
        self.m_showMenu = True
        self.menu.exec(global_)
        self.m_showMenu = False'''

    def focusInEvent(self, a0: QtGui.QFocusEvent):
        self.m_infocus = True
        p = self.parentWidget()
        p.installEventFilter(self)
        p.repaint()
        self.inFocus.emit(True)

    def focusOutEvent(self, a0: QtGui.QFocusEvent):
        if not self.m_isEditing:
            return
        if self.m_showMenu:
            return
        self.mode = Mode.NONE
        self.outFocus.emit(False)
        self.m_infocus = False

    def paintEvent(self, e: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        (r, g, b, a) = (255, 0, 0, 16)
        painter.fillRect(e.rect(), QColor(r, g, b, a))

        if self.m_infocus:
            rect = e.rect()
            rect.adjust(0, 0, -1, -1)
            painter.setPen(QColor(r, g, b))
            painter.drawRect(rect)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.position = QPoint(round(e.globalPosition().x() - self.geometry().x()),
                               round(e.globalPosition().y() - self.geometry().y()))
        if not self.m_isEditing:
            return
        if not self.m_infocus:
            return
        if not e.buttons() and QtCore.Qt.MouseButton.LeftButton:
            self.setCursorShape(e.pos())
            return
        if e.button() == QtCore.Qt.MouseButton.RightButton:
            # self.popupShow(e.pos())
            e.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if not self.m_isEditing: return
        if e.key() == QtCore.Qt.Key.Key_Delete:
            self.deleteLater()
        # Moving container with arrows
        if QApplication.keyboardModifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            newPos = QPoint(self.x(), self.y())
            if e.key() == QtCore.Qt.Key.Key_Up:
                newPos.setY(newPos.y() - 1)
            if e.key() == QtCore.Qt.Key.Key_Down:
                newPos.setY(newPos.y() + 1)
            if e.key() == QtCore.Qt.Key.Key_Left:
                newPos.setX(newPos.x() - 1)
            if e.key() == QtCore.Qt.Key.Key_Right:
                newPos.setX(newPos.x() + 1)
            self.move(newPos)

        if QApplication.keyboardModifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            if e.key() == QtCore.Qt.Key.Key_Up:
                self.resize(self.width(), self.height() - 1)
            if e.key() == QtCore.Qt.Key.Key_Down:
                self.resize(self.width(), self.height() + 1)
            if e.key() == QtCore.Qt.Key.Key_Left:
                self.resize(self.width() - 1, self.height())
            if e.key() == QtCore.Qt.Key.Key_Right:
                self.resize(self.width() + 1, self.height())
        self.newGeometry.emit(self.geometry())

    def setCursorShape(self, e_pos: QPoint):
        diff = 5
        # Left - Bottom

        if ((((e_pos.y() > self.y() + self.height() - diff) and  # Bottom
             (e_pos.x() < self.x() + diff)) or  # Left
                # Right-Bottom
                ((e_pos.y() > self.y() + self.height() - diff) and  # Bottom
                 (e_pos.x() > self.x() + self.width() - diff)) or  # Right
                # Left-Top
                ((e_pos.y() < self.y() + diff) and  # Top
                 (e_pos.x() < self.x() + diff)) or  # Left
                # Right-Top
                (e_pos.y() < self.y() + diff) and  # Top
                (e_pos.x() > self.x() + self.width() - diff))  # Right
                and self.m_isEditing):
            # Left - Bottom
            if ((e_pos.y() > self.y() + self.height() - diff) and  # Bottom
                    (e_pos.x() < self.x()
                     + diff)):  # Left
                self.mode = Mode.RESIZEBL
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeBDiagCursor))
                # Right - Bottom
            if ((e_pos.y() > self.y() + self.height() - diff) and  # Bottom
                    (e_pos.x() > self.x() + self.width() - diff)):  # Right
                self.mode = Mode.RESIZEBR
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeFDiagCursor))
            # Left - Top
            if ((e_pos.y() < self.y() + diff) and  # Top
                    (e_pos.x() < self.x() + diff)):  # Left
                self.mode = Mode.RESIZETL
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeFDiagCursor))
            # Right - Top
            if ((e_pos.y() < self.y() + diff) and  # Top
                    (e_pos.x() > self.x() + self.width() - diff)):  # Right
                self.mode = Mode.RESIZETR
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeBDiagCursor))
        # check cursor horizontal position
        elif ((e_pos.x() < self.x() + diff) or  # Left
              (e_pos.x() > self.x() + self.width() - diff)  # Right
              and self.m_isEditing):
            if e_pos.x() < self.x() + diff:  # Left
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeHorCursor))
                self.mode = Mode.RESIZEL
            else:  # Right
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeHorCursor))
                self.mode = Mode.RESIZER
        # check cursor vertical position
        elif ((e_pos.y() > self.y() + self.height() - diff) or  # Bottom
              (e_pos.y() < self.y() + diff)  # Top
              and self.m_isEditing):  # Top
            if e_pos.y() < self.y() + diff:  # Top
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeVerCursor))
                self.mode = Mode.RESIZET
            else:  # Bottom
                self.setCursor(QCursor(QtCore.Qt.CursorShape.SizeVerCursor))
                self.mode = Mode.RESIZEB
        else:
            self.setCursor(QCursor(QtCore.Qt.CursorShape.ArrowCursor))
            self.mode = Mode.MOVE

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        QWidget.mouseReleaseEvent(self, e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        QWidget.mouseMoveEvent(self, e)
        if not self.m_isEditing:
            return
        if not self.m_infocus:
            return
        if not e.buttons() and QtCore.Qt.MouseButton.LeftButton:
            p = QPoint(round(e.position().x() + self.geometry().x()), round(e.position().y() + self.geometry().y()))
            self.setCursorShape(p)
            return

        if (self.mode == Mode.MOVE or self.mode == Mode.NONE) and e.buttons() and QtCore.Qt.MouseButton.LeftButton:
            toMove = e.globalPosition().toPoint() - self.position
            if toMove.x() < 0:
                toMove.setX(0)
            if toMove.y() < 0:
                toMove.setY(0)
            if toMove.x() > self.parentWidget().width() - self.width():
                toMove.setX(self.parentWidget().width() - self.width())
            if toMove.y() > self.parentWidget().height() - self.height():
                toMove.setY(self.parentWidget().height() - self.height())
            self.move(toMove)
            self.newGeometry.emit(self.geometry())
            self.parentWidget().repaint()
            return
        if (self.mode != Mode.MOVE) and e.buttons() and QtCore.Qt.MouseButton.LeftButton:
            if self.mode == Mode.RESIZETL:  # Left - Top
                toMove = e.globalPosition().toPoint() - self.position
                if toMove.x() < 0:
                    toMove.setX(0)
                if toMove.y() < 0:
                    toMove.setY(0)
                newwidth = self.geometry().x() - toMove.x()
                newheight = self.geometry().y() - toMove.y()
                logger.debug(f'e.globalpos: {e.globalPosition().x()}\t'
                             f'self.position: {self.position.x()}\t'
                             f'self.geometry: {self.geometry().x()}\t'
                             f'self.geom.width: {self.geometry().width()}')
                self.resize(self.geometry().width() + newwidth, self.geometry().height() + newheight)
                self.move(toMove.x(), toMove.y())
            elif self.mode == Mode.RESIZETR:  # Right - Top
                toMove = e.globalPosition().toPoint() - self.position
                if toMove.y() < 0:
                    toMove.setY(0)
                newheight = self.geometry().y() - toMove.y()
                self.resize(e.position().x(), self.geometry().height() + newheight)
                self.move(self.x(), toMove.y())
            elif self.mode == Mode.RESIZEBL:  # Left - Bottom
                toMove = e.globalPosition().toPoint() - self.position
                if toMove.x() < 0:
                    toMove.setX(0)
                newwidth = self.geometry().x() - toMove.x()
                self.resize(self.geometry().width() + newwidth, e.position().y())
                self.move(toMove.x(), self.y())
            elif self.mode == Mode.RESIZEB:  # Bottom
                self.resize(self.width(), min(e.position().y(), self.parent().height() - self.y()))
            elif self.mode == Mode.RESIZEL:  # Left
                toMove = e.globalPosition().toPoint() - self.position
                if toMove.x() < 0:
                    toMove.setX(0)
                newwidth = self.geometry().x() - toMove.x()
                self.resize(self.geometry().width() + newwidth, self.height())
                self.move(toMove.x(), self.y())
            elif self.mode == Mode.RESIZET:  # Top
                toMove = e.globalPosition().toPoint() - self.position
                if toMove.y() < 0:
                    toMove.setY(0)
                newheight = self.geometry().y() - toMove.y()
                self.resize(self.width(), self.geometry().height() + newheight)
                self.move(self.x(), toMove.y())
            elif self.mode == Mode.RESIZER:  # Right
                self.resize(min(e.position().x(), self.parent().width() - self.x()), self.height())
            elif self.mode == Mode.RESIZEBR:  # Right - Bottom
                self.resize(min(e.position().x(), self.parent().width() - self.x()),
                            min(e.position().y(), self.parent().height() - self.y()))
            self.parentWidget().repaint()
        self.newGeometry.emit(self.geometry())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.showMaximized()

        self.widget = QWidget()
        self.widget.setLayout(QVBoxLayout())
        self.button = QPushButton()
        self.widget.layout().addWidget(self.button)
        self.button.setFixedSize(300, 300)
        self.label = QLabel('Label3')
        self.widget.layout().addWidget(self.label)
        self.label.setFixedSize(300, 600)
        self.setCentralWidget(self.widget)

        lab1 = QLabel("Label1")
        lab2 = QPushButton("Label2")
        con1 = MovableWidget(self.button, QPoint(10, 10), lab1)
        con2 = MovableWidget(self.label, QPoint(20, 50), lab2)

        con1.newGeometry.connect(lambda x: print(x))
        con2.newGeometry.connect(lambda x: print(x))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec()
