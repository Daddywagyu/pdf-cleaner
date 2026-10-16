"""Zoom / pan image viewer with mask overlay, brush tool and before/after slider."""
from __future__ import annotations

import numpy as np
import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView)


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def mask_to_qimage(mask: np.ndarray, color=(0, 220, 60), alpha=120) -> QImage:
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask > 0] = (*color, alpha)
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class _ClipItem(QGraphicsPixmapItem):
    """Pixmap drawn only to the right of clip_x (item coordinates)."""

    def __init__(self):
        super().__init__()
        self.clip_x: float | None = None

    def paint(self, painter, option, widget=None):
        if self.clip_x is None:
            return super().paint(painter, option, widget)
        painter.save()
        painter.setClipRect(QRectF(self.clip_x, -1e6, 2e6, 4e6))
        super().paint(painter, option, widget)
        painter.restore()


class PdfViewer(QGraphicsView):
    brushStarted = Signal()
    brushMoved = Signal(QPointF, QPointF)
    brushEnded = Signal()
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base, self._over = QGraphicsPixmapItem(), _ClipItem()
        self._mask = QGraphicsPixmapItem()
        self._line = QGraphicsLineItem()
        self._line.setPen(QPen(QColor(255, 255, 255), 2))
        self._line.setVisible(False)
        self._cursor = QGraphicsEllipseItem()
        self._cursor.setPen(QPen(QColor(255, 200, 0), 0))
        self._cursor.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._cursor.setVisible(False)
        for z, it in enumerate((self._base, self._over, self._mask, self._line, self._cursor)):
            it.setZValue(z)
            self._scene.addItem(it)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(58, 58, 62))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.tool = "navigate"            # navigate | add | restore
        self.brush_radius = 12
        self._painting = False
        self._pan_last = None
        self._slider_on = False
        self._split_x = 0.0
        self._drag_split = False
        self._size = (0, 0)
        self.horizontalScrollBar().valueChanged.connect(lambda _=0: self.viewChanged.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _=0: self.viewChanged.emit())

    # ---- content ---------------------------------------------------
    def set_base(self, bgr: np.ndarray | None):
        if bgr is None:
            self._base.setPixmap(QPixmap())
            return
        self._base.setPixmap(QPixmap.fromImage(bgr_to_qimage(bgr)))
        h, w = bgr.shape[:2]
        if (w, h) != self._size:
            self._size = (w, h)
            self._scene.setSceneRect(QRectF(0, 0, w, h))
            self._split_x = w / 2
            self.fit()

    def set_over(self, bgr: np.ndarray | None):
        self._over.setPixmap(QPixmap() if bgr is None else QPixmap.fromImage(bgr_to_qimage(bgr)))

    def set_mask(self, mask: np.ndarray | None, visible: bool = True):
        if mask is None or not visible:
            self._mask.setVisible(False)
            return
        self._mask.setPixmap(QPixmap.fromImage(mask_to_qimage(mask)))
        self._mask.setVisible(True)

    def set_slider(self, enabled: bool):
        self._slider_on = enabled
        self._over.clip_x = self._split_x if enabled else None
        self._over.setVisible(enabled)
        self._update_line()

    def _update_line(self):
        w, h = self._size
        self._line.setLine(self._split_x, 0, self._split_x, h)
        self._line.setVisible(self._slider_on)
        self._over.clip_x = self._split_x if self._slider_on else None
        self._over.update()

    def set_tool(self, tool: str):
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool == "navigate"
                         else QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if tool == "navigate"
                                  else Qt.CursorShape.CrossCursor)
        self._cursor.setVisible(False)

    def set_brush_radius(self, r: int):
        self.brush_radius = r

    # ---- zoom ------------------------------------------------------
    def fit(self):
        if self._size[0]:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.viewChanged.emit()

    def zoom(self, factor: float):
        self.scale(factor, factor)
        self.viewChanged.emit()

    def wheelEvent(self, e):
        self.zoom(1.25 if e.angleDelta().y() > 0 else 0.8)

    # ---- mouse -----------------------------------------------------
    def mousePressEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = e.position()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            if self._slider_on and abs(p.x() - self._split_x) < 14 / max(self.transform().m11(), 1e-6):
                self._drag_split = True
                return
            if self.tool != "navigate":
                self._painting = True
                self.brushStarted.emit()
                self._last = p
                self.brushMoved.emit(p, p)
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        if self._pan_last is not None:
            d = e.position() - self._pan_last
            self._pan_last = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._drag_split:
            self._split_x = float(np.clip(p.x(), 0, self._size[0]))
            self._update_line()
            return
        if self.tool != "navigate":
            r = self.brush_radius
            self._cursor.setRect(p.x() - r, p.y() - r, 2 * r, 2 * r)
            self._cursor.setVisible(True)
            if self._painting:
                self.brushMoved.emit(self._last, p)
                self._last = p
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = None
            return
        if self._drag_split:
            self._drag_split = False
            return
        if self._painting:
            self._painting = False
            self.brushEnded.emit()
            return
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        self._cursor.setVisible(False)
        super().leaveEvent(e)
