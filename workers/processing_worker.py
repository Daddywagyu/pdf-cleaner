"""Background QThreads (UI never blocks).  Each worker opens its own PdfLoader
(PyMuPDF documents must not be shared across threads) and supports safe cancel."""
from __future__ import annotations

import gc
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from config import Settings
from core import pdf_exporter, processing_pipeline as pp
from core.mask_editor import MaskEditor
from core.pdf_loader import PdfLoader


class ProcessingWorker(QThread):
    progress = Signal(int, int, str)          # done, total, message
    pageDone = Signal(int, str, object, dict) # index, restored png path ('' for detect), auto mask, info
    failed = Signal(str)
    cancelled = Signal()
    completed = Signal()

    def __init__(self, pdf_path: str, password: str | None, indices: list[int],
                 settings: Settings, edits: dict, tmp_dir: str, detect_only: bool = False, parent=None):
        super().__init__(parent)
        self.pdf_path, self.password = pdf_path, password
        self.indices, self.settings = list(indices), Settings(**settings.to_dict())
        self.edits, self.tmp_dir, self.detect_only = edits, tmp_dir, detect_only
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self):
        try:
            with PdfLoader(self.pdf_path, self.password) as loader:
                total = len(self.indices)
                for n, idx in enumerate(self.indices):
                    if self.is_cancelled():
                        self.cancelled.emit()
                        return
                    self.progress.emit(n, total, f"หน้า {idx + 1}/{loader.page_count}: เรนเดอร์")
                    img, used = loader.render(idx, self.settings.process_dpi)
                    s = Settings(**self.settings.to_dict())
                    s.process_dpi = used
                    self.progress.emit(n, total, f"หน้า {idx + 1}: ตรวจจับรอยปากกา")
                    if self.detect_only:
                        auto = pp.detect_only(img, s)
                        self.pageDone.emit(idx, "", auto, {"dpi": used})
                    else:
                        editor = None
                        if idx in self.edits:
                            editor = MaskEditor(img.shape[:2])
                            editor.set_layers(*self.edits[idx])
                        self.progress.emit(n, total, f"หน้า {idx + 1}: ฟื้นฟูพื้นหลัง")
                        res = pp.process_page(img, s, editor, cancel=self.is_cancelled)
                        if self.is_cancelled():
                            self.cancelled.emit()
                            return
                        path = str(Path(self.tmp_dir) / f"page_{idx}.png")
                        if not cv2.imwrite(path, res.restored):
                            raise IOError(f"เขียนไฟล์ชั่วคราวไม่สำเร็จ: {path}")
                        info = {"dpi": used, "residual": res.residual_ratio, "methods": res.methods,
                                "mask_px": int((res.final_mask > 0).sum())}
                        self.pageDone.emit(idx, path, res.auto_mask, info)
                    del img
                    gc.collect()
                    self.progress.emit(n + 1, total, f"หน้า {idx + 1}: เสร็จ")
            self.completed.emit()
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class ExportWorker(QThread):
    progress = Signal(int, int)
    failed = Signal(str)
    cancelled = Signal()
    done = Signal(object)

    def __init__(self, src, out, page_images, settings, password=None, parent=None):
        super().__init__(parent)
        self.args = (src, out, dict(page_images), Settings(**settings.to_dict()), password)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        src, out, imgs, settings, pw = self.args
        try:
            rep = pdf_exporter.export_pdf(src, out, imgs, settings, pw,
                                          progress=lambda a, b: self.progress.emit(a, b),
                                          cancel=self._cancel.is_set)
            self.done.emit(rep)
        except pdf_exporter.ExportError as e:
            if self._cancel.is_set():
                Path(out).unlink(missing_ok=True)
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
