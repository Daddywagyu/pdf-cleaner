"""Before/after comparison: Original | Processed | Split | Slider + brush tools."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget)

from gui.pdf_viewer import PdfViewer


class PreviewWidget(QWidget):
    brushStarted = Signal()
    brushMoved = Signal(QPointF, QPointF)
    brushEnded = Signal()
    undoRequested = Signal()
    redoRequested = Signal()
    resetRequested = Signal()
    maskVisibilityChanged = Signal(bool)

    VIEW_MODES = [("ต้นฉบับ", "original"), ("ผลลัพธ์", "processed"),
                  ("แบ่งหน้าจอ", "split"), ("สไลเดอร์ ก่อน/หลัง", "slider")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig = self.processed = self.mask = None
        self.left, self.right = PdfViewer(), PdfViewer()
        self.right.setVisible(False)
        self._sync_guard = False

        bar = QHBoxLayout()
        self.mode = QComboBox()
        for label, key in self.VIEW_MODES:
            self.mode.addItem(label, key)
        self.show_mask = QCheckBox("แสดงบริเวณที่จะลบ (Mask)")
        self.show_mask.setChecked(True)
        bar.addWidget(QLabel("มุมมอง:"))
        bar.addWidget(self.mode)
        bar.addWidget(self.show_mask)
        bar.addSpacing(12)
        self.tools = QButtonGroup(self)
        for text, key, tip in (("✋ เลื่อน", "navigate", "เลื่อน/ซูมภาพ (เมาส์กลางลากเพื่อเลื่อนได้ทุกโหมด)"),
                               ("🖌 เพิ่มพื้นที่ลบ", "add", "ระบายเพื่อเพิ่มพื้นที่ที่ต้องการลบ"),
                               ("🧽 คืนพื้นที่", "restore", "ระบายเพื่อคืนพื้นที่ที่ระบบตรวจผิด")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setProperty("tool", key)
            self.tools.addButton(b)
            bar.addWidget(b)
            if key == "navigate":
                b.setChecked(True)
        bar.addWidget(QLabel("ขนาดแปรง"))
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(2, 80)
        self.brush.setValue(12)
        self.brush.setFixedWidth(90)
        bar.addWidget(self.brush)
        self.undo_b, self.redo_b, self.reset_b = QPushButton("↶ Undo"), QPushButton("↷ Redo"), QPushButton("Reset หน้า")
        for b in (self.undo_b, self.redo_b, self.reset_b):
            bar.addWidget(b)
        zin, zout, fit = QPushButton("＋"), QPushButton("－"), QPushButton("พอดีหน้าจอ")
        for b in (zout, zin, fit):
            bar.addWidget(b)
        bar.addStretch(1)

        views = QHBoxLayout()
        views.addWidget(self.left, 1)
        views.addWidget(self.right, 1)
        self.hint = QLabel("")
        self.hint.setStyleSheet("color:#888;")
        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addLayout(views, 1)
        lay.addWidget(self.hint)

        self.mode.currentIndexChanged.connect(self.refresh)
        self.show_mask.toggled.connect(lambda v: (self.refresh(), self.maskVisibilityChanged.emit(v)))
        self.tools.buttonClicked.connect(self._tool_changed)
        self.brush.valueChanged.connect(self._brush_changed)
        self.undo_b.clicked.connect(self.undoRequested)
        self.redo_b.clicked.connect(self.redoRequested)
        self.reset_b.clicked.connect(self.resetRequested)
        zin.clicked.connect(lambda: self._zoom(1.25))
        zout.clicked.connect(lambda: self._zoom(0.8))
        fit.clicked.connect(lambda: (self.left.fit(), self.right.fit()))
        for v in (self.left, self.right):
            v.brushStarted.connect(self.brushStarted)
            v.brushMoved.connect(self.brushMoved)
            v.brushEnded.connect(self.brushEnded)
            v.viewChanged.connect(lambda v=v: self._sync(v))

    # ---- API ---------------------------------------------------------
    def set_images(self, orig: np.ndarray | None, processed: np.ndarray | None, mask: np.ndarray | None):
        self.orig, self.processed, self.mask = orig, processed, mask
        self.refresh()

    def set_mask(self, mask: np.ndarray | None):
        self.mask = mask
        self.left.set_mask(mask, self.show_mask.isChecked())
        self.right.set_mask(None, False)

    def set_undo_state(self, can_undo: bool, can_redo: bool):
        self.undo_b.setEnabled(can_undo)
        self.redo_b.setEnabled(can_redo)

    # ---- internals -----------------------------------------------------
    def refresh(self):
        mode = self.mode.currentData()
        o, p = self.orig, self.processed
        self.right.setVisible(mode == "split")
        self.hint.setText("" if p is not None or mode == "original"
                          else "ยังไม่มีผลลัพธ์ — กด 'ประมวลผลหน้านี้' ก่อน")
        self.left.set_over(None)
        self.left.set_slider(False)
        if mode == "original" or o is None:
            self.left.set_base(o)
        elif mode == "processed":
            self.left.set_base(p if p is not None else o)
        elif mode == "split":
            self.left.set_base(o)
            self.right.set_base(p if p is not None else o)
        elif mode == "slider":
            self.left.set_base(o)
            self.left.set_over(p if p is not None else o)
            self.left.set_slider(True)
        self.left.set_mask(self.mask, self.show_mask.isChecked())
        self.right.set_mask(None, False)

    def _tool_changed(self, btn):
        key = btn.property("tool")
        self.left.set_tool(key)
        self.right.set_tool("navigate")

    def _brush_changed(self, v):
        self.left.set_brush_radius(v)

    def _zoom(self, f):
        self.left.zoom(f)

    def _sync(self, src: PdfViewer):
        if self._sync_guard or not self.right.isVisible():
            return
        dst = self.right if src is self.left else self.left
        self._sync_guard = True
        try:
            dst.setTransform(src.transform())
            dst.horizontalScrollBar().setValue(src.horizontalScrollBar().value())
            dst.verticalScrollBar().setValue(src.verticalScrollBar().value())
        finally:
            self._sync_guard = False
