"""Main window: wires loader, preview, settings and background workers."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from config import APP_NAME, Settings
from core import pdf_exporter
from core.mask_editor import MaskEditor
from core.pdf_loader import PdfLoader, PdfOpenError, PdfPasswordRequired
from gui.preview_widget import PreviewWidget
from gui.settings_panel import SettingsPanel
from workers.processing_worker import ExportWorker, ProcessingWorker

log = logging.getLogger(__name__)


@dataclass
class PageState:
    auto_mask: np.ndarray | None = None
    editor: MaskEditor | None = None
    processed_path: str | None = None
    stale: bool = False
    info: dict = field(default_factory=dict)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1450, 900)
        self.setAcceptDrops(True)
        self.loader: PdfLoader | None = None
        self.pdf_path: str | None = None
        self.password: str | None = None
        self.pages: dict[int, PageState] = {}
        self.current = -1
        self.orig_img: np.ndarray | None = None
        self.processed_img: np.ndarray | None = None
        self.worker = None
        self._tmp = tempfile.TemporaryDirectory(prefix="penremover_")
        self._thumb_queue: list[int] = []
        self._last_settings: Settings | None = None

        # --- widgets ---
        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbs.setIconSize(QSize(120, 160))
        self.thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbs.setMovement(QListWidget.Movement.Static)
        self.thumbs.setFixedWidth(170)
        self.thumbs.setSpacing(6)
        self.thumbs.currentRowChanged.connect(self.show_page)

        self.preview = PreviewWidget()
        self.panel = SettingsPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setFixedWidth(330)

        self.open_btn = QPushButton("📂 Open PDF")
        self.open_btn.clicked.connect(self.open_dialog)
        self.prev_btn, self.next_btn = QPushButton("◀"), QPushButton("▶")
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setEnabled(False)
        self.info_lbl = QLabel("ลากไฟล์ PDF มาวางในหน้าต่างนี้ หรือกด Open PDF")
        top = QHBoxLayout()
        for w in (self.open_btn, self.prev_btn, self.page_spin, self.next_btn):
            top.addWidget(w)
        top.addWidget(self.info_lbl, 1)
        self.prev_btn.clicked.connect(lambda: self.goto(self.current - 1))
        self.next_btn.clicked.connect(lambda: self.goto(self.current + 1))
        self.page_spin.editingFinished.connect(lambda: self.goto(self.page_spin.value() - 1))

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.addLayout(top)
        cl.addWidget(self.preview, 1)
        cl.addWidget(self.progress)
        cl.addWidget(self.log_box)
        split = QSplitter()
        split.addWidget(self.thumbs)
        split.addWidget(center)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        # --- actions / signals ---
        act = QAction("Open PDF…", self)
        act.setShortcut(QKeySequence.StandardKey.Open)
        act.triggered.connect(self.open_dialog)
        self.menuBar().addMenu("ไฟล์").addAction(act)
        for text, seq, fn in (("Undo", QKeySequence.StandardKey.Undo, self.undo),
                              ("Redo", QKeySequence.StandardKey.Redo, self.redo)):
            a = QAction(text, self)
            a.setShortcut(seq)
            a.triggered.connect(fn)
            self.addAction(a)
        p = self.panel
        p.previewClicked.connect(self.run_preview)
        p.processCurrentClicked.connect(lambda: self.run_process([self.current]))
        p.processAllClicked.connect(self.run_process_all)
        p.exportClicked.connect(self.export)
        p.cancelClicked.connect(self.cancel_job)
        p.changed.connect(self.on_settings_changed)
        pv = self.preview
        pv.brushStarted.connect(self.on_brush_start)
        pv.brushMoved.connect(self.on_brush_move)
        pv.brushEnded.connect(self.on_brush_end)
        pv.undoRequested.connect(self.undo)
        pv.redoRequested.connect(self.redo)
        pv.resetRequested.connect(self.reset_page)
        pv.set_undo_state(False, False)
        self.set_enabled(False)

    # ------------------------------------------------------------ utils
    def say(self, msg: str, error: bool = False):
        self.log_box.appendPlainText(("❌ " if error else "") + msg)
        (log.error if error else log.info)(msg)

    def settings(self) -> Settings:
        return self.panel.settings()

    def set_enabled(self, on: bool):
        for b in (self.panel.b_prev, self.panel.b_cur, self.panel.b_all, self.panel.b_exp):
            b.setEnabled(on)

    # ------------------------------------------------------- open files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(".pdf") for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.toLocalFile().lower().endswith(".pdf"):
                self.open_pdf(u.toLocalFile())
                break

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์ PDF", "", "PDF (*.pdf)")
        if path:
            self.open_pdf(path)

    def open_pdf(self, path: str, password: str | None = None):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, APP_NAME, "กำลังประมวลผลอยู่ กรุณายกเลิกงานก่อน")
            return
        try:
            loader = PdfLoader(path, password)
        except PdfPasswordRequired:
            pw, ok = QInputDialog.getText(self, "PDF มีรหัสผ่าน", "รหัสผ่าน:", QLineEdit.EchoMode.Password)
            if ok and pw:
                return self.open_pdf(path, pw)
            return
        except PdfOpenError as e:
            QMessageBox.critical(self, APP_NAME, str(e))
            self.say(str(e), True)
            return
        if self.loader:
            self.loader.close()
        self.loader, self.pdf_path, self.password = loader, path, password
        self.pages = {i: PageState() for i in range(loader.page_count)}
        self.current = -1
        w, h = loader.page_size(0)
        self.info_lbl.setText(f"{Path(path).name} — {loader.page_count} หน้า — หน้าแรก {w:.0f}×{h:.0f} pt "
                              f"({w / 72 * 25.4:.0f}×{h / 72 * 25.4:.0f} mm)")
        self.say(f"เปิดไฟล์: {path} ({loader.page_count} หน้า)")
        self.page_spin.setMaximum(loader.page_count)
        self.page_spin.setEnabled(True)
        self.thumbs.clear()
        for i in range(loader.page_count):
            self.thumbs.addItem(QListWidgetItem(f"{i + 1}"))
        self._thumb_queue = list(range(loader.page_count))
        QTimer.singleShot(0, self._next_thumb)
        self.set_enabled(True)
        self.thumbs.setCurrentRow(0)

    def _next_thumb(self):
        if not self.loader or not self._thumb_queue:
            return
        i = self._thumb_queue.pop(0)
        try:
            arr = self.loader.thumbnail(i, 160)
            h, w = arr.shape[:2]
            img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            it = self.thumbs.item(i)
            if it:
                it.setIcon(QIcon(QPixmap.fromImage(img)))
        except Exception as e:                                  # noqa: BLE001
            self.say(f"สร้าง Thumbnail หน้า {i + 1} ไม่ได้: {e}", True)
        QTimer.singleShot(0, self._next_thumb)

    # ---------------------------------------------------------- paging
    def goto(self, i: int):
        if self.loader and 0 <= i < self.loader.page_count:
            self.thumbs.setCurrentRow(i)

    def show_page(self, i: int):
        if not self.loader or i < 0 or i == self.current and self.orig_img is not None:
            return
        self.current = i
        self.page_spin.setValue(i + 1)
        try:
            self.orig_img, used = self.loader.render(i, self.settings().process_dpi)
        except Exception as e:                                  # noqa: BLE001
            self.say(f"เรนเดอร์หน้า {i + 1} ไม่ได้: {e}", True)
            return
        st = self.pages[i]
        self.processed_img = cv2.imread(st.processed_path) if st.processed_path else None
        if st.editor is None or st.editor.shape != self.orig_img.shape[:2]:
            st.editor = None
        self._refresh_preview(keep_view=False)
        w, h = self.loader.page_size(i)
        self.statusBar().showMessage(f"หน้า {i + 1}/{self.loader.page_count} — {w:.0f}×{h:.0f} pt — "
                                     f"เรนเดอร์ {used} DPI — "
                                     f"{'มีข้อความ' if self.loader.has_text(i) else 'ไม่มีข้อความ (ภาพ/สแกน)'}")

    def _final_mask(self, st: PageState):
        if st.auto_mask is None and (st.editor is None or not st.editor.has_edits()):
            return None
        shape = self.orig_img.shape[:2]
        auto = st.auto_mask if st.auto_mask is not None and st.auto_mask.shape == shape else None
        if st.editor is not None:
            return st.editor.final_mask(auto)
        return auto

    def _refresh_preview(self, keep_view=True):
        st = self.pages[self.current]
        self.preview.set_images(self.orig_img, self.processed_img, self._final_mask(st))
        e = st.editor
        self.preview.set_undo_state(bool(e and e.can_undo), bool(e and e.can_redo))

    # --------------------------------------------------------- settings
    def on_settings_changed(self):
        s = self.settings()
        old = self._last_settings
        self._last_settings = s
        if old is None or not self.loader:
            return
        if old.process_dpi != s.process_dpi:
            for st in self.pages.values():
                st.auto_mask, st.editor, st.processed_path, st.stale = None, None, None, False
            self.say("เปลี่ยน DPI: ล้างผลตรวจจับ/แก้ไขเดิมของทุกหน้า (ความละเอียด Mask เปลี่ยน)")
            self.current = -1
            self.show_page(self.thumbs.currentRow())
        else:
            for st in self.pages.values():
                st.auto_mask = None
                st.stale = st.processed_path is not None
            if self.orig_img is not None:
                self._refresh_preview()
            self.say("เปลี่ยนการตั้งค่า: กด Preview เพื่อคำนวณ Mask ใหม่")

    # ---------------------------------------------------------- brush
    def _ensure_editor(self) -> MaskEditor | None:
        if self.orig_img is None:
            return None
        st = self.pages[self.current]
        if st.editor is None:
            st.editor = MaskEditor(self.orig_img.shape[:2])
        return st.editor

    def on_brush_start(self):
        e = self._ensure_editor()
        if e:
            e.begin_stroke()

    def on_brush_move(self, p0: QPointF, p1: QPointF):
        e = self._ensure_editor()
        if not e:
            return
        mode = self.preview.left.tool
        e.paint((p0.x(), p0.y()), (p1.x(), p1.y()), self.preview.left.brush_radius, mode)
        st = self.pages[self.current]
        self.preview.set_mask(self._final_mask(st))

    def on_brush_end(self):
        st = self.pages[self.current]
        st.stale = st.processed_path is not None
        if st.stale:
            self.say("แก้ไข Mask แล้ว — กด Process Current Page เพื่ออัปเดตผลลัพธ์")
        self._refresh_preview()

    def undo(self):
        st = self.pages.get(self.current)
        if st and st.editor and st.editor.undo():
            st.stale = st.processed_path is not None
            self._refresh_preview()

    def redo(self):
        st = self.pages.get(self.current)
        if st and st.editor and st.editor.redo():
            st.stale = st.processed_path is not None
            self._refresh_preview()

    def reset_page(self):
        if self.current < 0:
            return
        if QMessageBox.question(self, APP_NAME, "รีเซ็ตหน้านี้กลับเป็นภาพต้นฉบับ (ล้าง Mask, การแก้ไข และผลลัพธ์)?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.pages[self.current] = PageState()
        self.processed_img = None
        self._refresh_preview()
        self.say(f"รีเซ็ตหน้า {self.current + 1} แล้ว")

    # ------------------------------------------------------------ jobs
    def _busy(self) -> bool:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, APP_NAME, "มีงานกำลังทำอยู่ กรุณารอหรือกดยกเลิก")
            return True
        return False

    def _edits_for(self, indices):
        out = {}
        for i in indices:
            e = self.pages[i].editor
            if e is not None and e.has_edits():
                out[i] = e.layers()
        return out

    def _start(self, worker: ProcessingWorker):
        self.worker = worker
        self.panel.set_busy(True)
        self.progress.setValue(0)
        worker.progress.connect(self.on_progress)
        worker.pageDone.connect(self.on_page_done)
        worker.failed.connect(self.on_failed)
        worker.cancelled.connect(lambda: self._job_end("ยกเลิกงานแล้ว"))
        worker.completed.connect(lambda: self._job_end("เสร็จสิ้น"))
        worker.start()

    def run_preview(self):
        if self.current < 0 or self._busy():
            return
        self._start(ProcessingWorker(self.pdf_path, self.password, [self.current], self.settings(),
                                     {}, self._tmp.name, detect_only=True, parent=self))

    def run_process(self, indices):
        if not self.loader or self._busy() or not indices or indices[0] < 0:
            return
        self._start(ProcessingWorker(self.pdf_path, self.password, indices, self.settings(),
                                     self._edits_for(indices), self._tmp.name, parent=self))

    def run_process_all(self):
        if not self.loader:
            return
        n = self.loader.page_count
        if QMessageBox.question(self, APP_NAME, f"ประมวลผลทั้ง {n} หน้า? (ตรวจสอบหน้าที่ต้องการด้วย Preview ก่อนได้)") \
                == QMessageBox.StandardButton.Yes:
            self.run_process(list(range(n)))

    def cancel_job(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.say("กำลังยกเลิก…")

    def on_progress(self, done, total, msg):
        self.progress.setValue(int(100 * done / max(1, total)))
        self.statusBar().showMessage(msg)

    def on_page_done(self, idx, path, auto, info):
        st = self.pages[idx]
        st.auto_mask = auto
        if path:
            st.processed_path, st.stale, st.info = path, False, info
            m = ", ".join(f"{k}:{v}" for k, v in info.get("methods", {}).items())
            self.say(f"หน้า {idx + 1}: ลบแล้ว ({info.get('mask_px', 0):,} px; วิธี {m}; "
                     f"สีตกค้าง {info.get('residual', 0) * 100:.1f}%)")
            it = self.thumbs.item(idx)
            if it:
                it.setText(f"{idx + 1} ✓")
        else:
            self.say(f"หน้า {idx + 1}: ตรวจจับ Mask แล้ว ({int((auto > 0).sum()):,} px) — ตรวจสอบสีเขียวบนภาพ")
        if idx == self.current:
            self.processed_img = cv2.imread(st.processed_path) if st.processed_path else self.processed_img
            self._refresh_preview()

    def on_failed(self, msg):
        self.say(msg, True)
        QMessageBox.critical(self, APP_NAME, f"เกิดข้อผิดพลาด:\n{msg}")
        self._job_end("ล้มเหลว")

    def _job_end(self, msg):
        self.panel.set_busy(False)
        self.set_enabled(self.loader is not None)
        self.progress.setValue(100 if msg == "เสร็จสิ้น" else 0)
        self.say(msg)

    # ----------------------------------------------------------- export
    def export(self):
        if not self.loader or self._busy():
            return
        imgs = {i: s.processed_path for i, s in self.pages.items() if s.processed_path}
        stale = [i + 1 for i, s in self.pages.items() if s.stale]
        if not imgs:
            QMessageBox.information(self, APP_NAME, "ยังไม่มีหน้าที่ประมวลผล กด Process Current/All Pages ก่อน")
            return
        if stale and QMessageBox.question(self, APP_NAME, f"หน้า {stale} มีการแก้ไข Mask ที่ยังไม่ประมวลผลใหม่ ส่งออกต่อไหม?") \
                != QMessageBox.StandardButton.Yes:
            return
        s = self.settings()
        has_text = [i + 1 for i in imgs if self.loader.has_text(i)]
        if has_text:
            note = ("จะเก็บข้อความเป็น Text Layer ที่มองไม่เห็น (เฉพาะอักขระ Latin-1; ภาษาไทยจะค้นหา/คัดลอกไม่ได้)"
                    if s.keep_text_layer else "ข้อความจะถูกแปลงเป็นภาพ ค้นหา/คัดลอกไม่ได้")
            if QMessageBox.question(self, APP_NAME, f"หน้า {has_text} มีข้อความจริง/เวกเตอร์ ซึ่งจะถูกเรนเดอร์เป็นภาพ\n{note}\nดำเนินการต่อ?") \
                    != QMessageBox.StandardButton.Yes:
                return
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ปลายทาง", str(Path(self.pdf_path).parent))
        if not folder:
            return
        out = pdf_exporter.default_output_name(self.pdf_path, folder)
        if out.exists() and QMessageBox.question(self, APP_NAME, f"{out.name} มีอยู่แล้ว เขียนทับไฟล์ผลลัพธ์เดิม?") \
                != QMessageBox.StandardButton.Yes:
            return
        w = ExportWorker(self.pdf_path, str(out), imgs, s, self.password, parent=self)
        self.worker = w
        self.panel.set_busy(True)
        w.progress.connect(lambda a, b: self.progress.setValue(int(100 * a / max(1, b))))
        w.failed.connect(self.on_failed)
        w.cancelled.connect(lambda: self._job_end("ยกเลิกการส่งออก"))
        w.done.connect(self.on_export_done)
        w.start()

    def on_export_done(self, rep):
        self._job_end("เสร็จสิ้น")
        txt = f"บันทึกแล้ว: {rep.output}\nจำนวนหน้า {rep.pages_total} (แทนที่ {len(rep.pages_replaced)} หน้า) — ตรวจสอบไฟล์: " \
              f"{'ผ่าน' if rep.verified else 'ไม่ผ่าน'}"
        if rep.warnings:
            txt += "\n\nคำเตือน:\n" + "\n".join(rep.warnings[:10])
        self.say(txt)
        QMessageBox.information(self, APP_NAME, txt)      # ไม่เปิดไฟล์ด้วยโปรแกรมภายนอกอัตโนมัติ

    # ------------------------------------------------------------ close
    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        if self.loader:
            self.loader.close()
        self._tmp.cleanup()
        super().closeEvent(e)
