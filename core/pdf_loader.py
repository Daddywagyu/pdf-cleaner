"""PDF reading / rendering (PyMuPDF).  One loader instance = one open document;
loaders are NOT shared between threads (each worker opens its own)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf

from config import MAX_PIXELS

pymupdf.TOOLS.mupdf_display_errors(False)


class PdfOpenError(Exception):
    """Raised when a PDF cannot be opened (message is user-presentable, Thai)."""


class PdfPasswordRequired(PdfOpenError):
    pass


def safe_dpi(width_pt: float, height_pt: float, dpi: int, max_pixels: int = MAX_PIXELS) -> int:
    """Lower the DPI when the page would exceed the memory budget."""
    px = (width_pt * dpi / 72.0) * (height_pt * dpi / 72.0)
    if px <= max_pixels:
        return dpi
    return max(72, int(dpi * (max_pixels / px) ** 0.5))


class PdfLoader:
    def __init__(self, path: str | Path, password: str | None = None):
        self.path = str(path)
        self.doc: pymupdf.Document | None = None
        p = Path(self.path)
        if not p.is_file():
            raise PdfOpenError(f"ไม่พบไฟล์: {p}")
        try:
            doc = pymupdf.open(self.path)
        except Exception as e:                                  # noqa: BLE001
            raise PdfOpenError(f"ไฟล์ PDF เสียหายหรือเปิดไม่ได้: {e}") from e
        if doc.needs_pass:
            if not password or not doc.authenticate(password):
                doc.close()
                raise PdfPasswordRequired("ไฟล์ PDF มีรหัสผ่าน")
        if doc.page_count == 0:
            doc.close()
            raise PdfOpenError("ไฟล์ PDF ไม่มีหน้า")
        try:
            doc.load_page(0)
        except Exception as e:                                  # noqa: BLE001
            doc.close()
            raise PdfOpenError(f"ไม่สามารถประมวลผลไฟล์นี้ได้: {e}") from e
        self.doc = doc
        self.password = password

    # -- info ---------------------------------------------------------
    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def page_size(self, index: int) -> tuple[float, float]:
        r = self.doc[index].rect
        return float(r.width), float(r.height)

    def has_text(self, index: int) -> bool:
        return len(self.doc[index].get_text("text").strip()) > 0

    def has_vector_or_text(self, index: int) -> bool:
        pg = self.doc[index]
        return self.has_text(index) or len(pg.get_drawings()) > 0

    # -- rendering ----------------------------------------------------
    def render(self, index: int, dpi: int) -> tuple[np.ndarray, int]:
        """Return (BGR uint8 image, dpi actually used)."""
        pg = self.doc[index]
        used = safe_dpi(pg.rect.width, pg.rect.height, dpi)
        z = used / 72.0
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False, colorspace=pymupdf.csRGB)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), used

    def thumbnail(self, index: int, max_side: int = 140) -> np.ndarray:
        pg = self.doc[index]
        z = max_side / max(pg.rect.width, pg.rect.height)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False, colorspace=pymupdf.csRGB)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        return arr.copy()                                       # RGB

    def close(self):
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
