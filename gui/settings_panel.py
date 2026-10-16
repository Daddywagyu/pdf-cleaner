"""Right-hand settings panel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from config import Settings


def _slider(lo, hi, val):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class SettingsPanel(QWidget):
    changed = Signal()
    previewClicked = Signal()
    processCurrentClicked = Signal()
    processAllClicked = Signal()
    exportClicked = Signal()
    cancelClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        d = Settings()
        lay = QVBoxLayout(self)

        g = QGroupBox("โหมดการลบ")
        f = QFormLayout(g)
        self.mode = QComboBox()
        self.mode.addItem("A: Conservative (ลบเฉพาะที่มั่นใจ)", "conservative")
        self.mode.addItem("B: Balanced (ค่าเริ่มต้น)", "balanced")
        self.mode.addItem("C: Advanced (วิเคราะห์ขั้นสูง)", "advanced")
        self.mode.setCurrentIndex(1)
        f.addRow(self.mode)
        lay.addWidget(g)

        g = QGroupBox("สีที่ตรวจจับ")
        f = QFormLayout(g)
        self.red = QCheckBox("ปากกาสีแดง")
        self.red.setChecked(True)
        self.sens_red = _slider(0, 100, int(d.sens_red * 100))
        self.blue = QCheckBox("ปากกาสีน้ำเงิน")
        self.blue.setChecked(True)
        self.sens_blue = _slider(0, 100, int(d.sens_blue * 100))
        f.addRow(self.red, self.sens_red)
        f.addRow(self.blue, self.sens_blue)
        f.addRow(QLabel("สไลเดอร์ = Detection Sensitivity (สูง = ไวขึ้น/จับสีอ่อนได้)"))
        lay.addWidget(g)

        g = QGroupBox("Mask / Restoration")
        f = QFormLayout(g)
        self.expand = _slider(0, 10, d.mask_expand)
        self.radius = _slider(1, 15, d.inpaint_radius)
        self.method = QComboBox()
        for label, key in (("อัตโนมัติ (เลือกตามพื้นที่)", "auto"), ("Inpaint: Telea", "telea"),
                           ("Inpaint: Navier-Stokes", "ns"), ("ประมาณพื้นหลัง", "background")):
            self.method.addItem(label, key)
        f.addRow("Mask Expansion", self.expand)
        f.addRow("Inpainting Radius", self.radius)
        f.addRow("วิธีฟื้นฟู", self.method)
        lay.addWidget(g)

        g = QGroupBox("ส่งออก")
        f = QFormLayout(g)
        self.dpi = QComboBox()
        for v in (200, 300, 400):
            self.dpi.addItem(f"{v} DPI", v)
        self.fmt = QComboBox()
        self.fmt.addItem("PNG (ไม่สูญเสียคุณภาพ)", "png")
        self.fmt.addItem("JPEG คุณภาพ 95", "jpeg")
        self.textlayer = QCheckBox("คงข้อความที่ค้นหาได้ (Text Layer)")
        self.textlayer.setChecked(True)
        f.addRow("DPI เรนเดอร์", self.dpi)
        f.addRow("รูปแบบภาพ", self.fmt)
        f.addRow(self.textlayer)
        lay.addWidget(g)

        self.b_prev = QPushButton("Preview (ตรวจจับ/แสดง Mask)")
        self.b_cur = QPushButton("Process Current Page")
        self.b_all = QPushButton("Process All Pages")
        self.b_exp = QPushButton("Export PDF")
        self.b_cancel = QPushButton("ยกเลิกงาน")
        self.b_cancel.setEnabled(False)
        for b, sig in ((self.b_prev, self.previewClicked), (self.b_cur, self.processCurrentClicked),
                       (self.b_all, self.processAllClicked), (self.b_exp, self.exportClicked),
                       (self.b_cancel, self.cancelClicked)):
            b.clicked.connect(sig)
            lay.addWidget(b)
        lay.addStretch(1)

        for w in (self.mode, self.method, self.dpi, self.fmt):
            w.currentIndexChanged.connect(lambda _=0: self.changed.emit())
        for w in (self.red, self.blue, self.textlayer):
            w.toggled.connect(lambda _=False: self.changed.emit())
        for w in (self.sens_red, self.sens_blue, self.expand, self.radius):
            w.sliderReleased.connect(self.changed)
            w.valueChanged.connect(lambda _=0, w=w: (not w.isSliderDown()) and self.changed.emit())

    def settings(self) -> Settings:
        return Settings(
            mode=self.mode.currentData(), detect_red=self.red.isChecked(), detect_blue=self.blue.isChecked(),
            sens_red=self.sens_red.value() / 100.0, sens_blue=self.sens_blue.value() / 100.0,
            mask_expand=self.expand.value(), inpaint_radius=self.radius.value(),
            restore_method=self.method.currentData(), process_dpi=self.dpi.currentData(),
            keep_text_layer=self.textlayer.isChecked(), image_format=self.fmt.currentData(),
        )

    def set_busy(self, busy: bool):
        for b in (self.b_prev, self.b_cur, self.b_all, self.b_exp):
            b.setEnabled(not busy)
        self.b_cancel.setEnabled(busy)
