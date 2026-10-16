"""Synthetic test documents with known ground truth (clean page + pen mask)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf

PAGE_W, PAGE_H = 595, 842            # A4 in points
LOREM = ("The quick brown fox jumps over the lazy dog. Quarterly revenue grew by "
         "twelve percent while operating costs remained stable across all regions.")


def build_clean_pdf(path: str | Path, pages: int = 1, rotate: int = 0) -> Path:
    """PDF with real (searchable) text, a table, a vector chart and a raster image."""
    doc = pymupdf.open()
    rng = np.random.default_rng(7)
    for p in range(pages):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text((50, 60), f"Annual Report - Page {p + 1}", fontsize=20, fontname="hebo")
        y = 95
        for i in range(14):
            page.insert_text((50, y), f"{i + 1}. {LOREM}"[:92], fontsize=10, fontname="helv")
            y += 17
        # table
        x0, y0 = 50, 360
        for r in range(6):
            page.draw_line((x0, y0 + r * 22), (x0 + 400, y0 + r * 22), color=(0, 0, 0), width=1)
        for c in range(5):
            page.draw_line((x0 + c * 100, y0), (x0 + c * 100, y0 + 110), color=(0, 0, 0), width=1)
        for r in range(5):
            for c in range(4):
                page.insert_text((x0 + c * 100 + 6, y0 + r * 22 + 15), f"{(r + 1) * (c + 3)}.5", fontsize=9)
        # printed coloured elements that must survive (context test)
        page.insert_text((50, 520), "Legend:", fontsize=10)
        page.insert_text((95, 520), "Revenue (blue)  Costs (red)  Margin (green)", fontsize=10,
                         color=(0.55, 0.0, 0.0))
        page.draw_rect(pymupdf.Rect(50, 530, 62, 542), color=(0, 0, 0.8), fill=(0, 0, 0.8))
        page.draw_rect(pymupdf.Rect(70, 530, 82, 542), color=(0.8, 0, 0), fill=(0.8, 0, 0))
        # raster picture (gradient + texture)
        img = np.zeros((120, 200, 3), np.uint8)
        img[..., 0] = np.linspace(40, 200, 200)[None, :]
        img[..., 1] = np.linspace(90, 160, 120)[:, None]
        img[..., 2] = 110
        img += rng.integers(0, 12, img.shape, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        page.insert_image(pymupdf.Rect(50, 580, 250, 700), stream=buf.tobytes())
        if rotate:
            page.set_rotation(rotate)
    doc.save(str(path))
    doc.close()
    return Path(path)


def render(path, index=0, dpi=200) -> np.ndarray:
    doc = pymupdf.open(str(path))
    pix = doc[index].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False,
                                colorspace=pymupdf.csRGB)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _stroke_alpha(shape, pts, thickness) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.polylines(m, [np.array(pts, np.int32)], False, 255, thickness, cv2.LINE_AA)
    return m.astype(np.float32) / 255.0


def add_pens(clean: np.ndarray, kind: str = "all", seed: int = 3):
    """Return (dirty, pen_mask_gt).  Pens: opaque red, opaque blue, light translucent."""
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    s = w / 1654.0
    dirty = clean.astype(np.float32).copy()
    gt = np.zeros((h, w), np.uint8)

    def P(x, y):                                  # coordinates at 200 DPI A4 for readability
        return (int(x * s), int(y * s))

    strokes = []
    if kind in ("all", "red"):
        # red scribble over text paragraph (opaque)
        strokes.append(("opaque", (30, 30, 200), [P(150, 300), P(700, 330), P(250, 360), P(900, 390)], 9))
        # red circle over the table and a line across table cells
        strokes.append(("opaque", (35, 35, 210), [P(300, 1000), P(900, 1010), P(920, 1150), P(300, 1140), P(300, 1000)], 8))
    if kind in ("all", "blue"):
        strokes.append(("opaque", (190, 60, 25), [P(200, 560), P(1100, 580)], 10))
        strokes.append(("opaque", (200, 70, 30), [P(300, 1500), P(450, 1580), P(600, 1510)], 9))  # just above the photo
    if kind in ("all", "light"):
        # translucent highlighter-like strokes (multiplicative) over text
        strokes.append(("translucent", (0.86, 0.68, 1.0), [P(140, 450), P(1000, 452)], 26))
        strokes.append(("translucent", (1.0, 0.82, 0.80), [P(140, 520), P(900, 522)], 26))
    for typ, col, pts, th in strokes:
        a = _stroke_alpha(clean.shape, pts, max(1, int(th * s)))
        if typ == "opaque":
            for c in range(3):
                dirty[..., c] = dirty[..., c] * (1 - a) + col[c] * a
        else:
            for c in range(3):
                dirty[..., c] *= (1 - a) + col[c] * a
        gt |= (a > 0.5).astype(np.uint8) * 255
    dirty += rng.normal(0, 1.2, dirty.shape).astype(np.float32)
    return np.clip(dirty, 0, 255).astype(np.uint8), gt


def build_samples(out_dir: str | Path) -> list[Path]:
    """Create the sample PDFs shipped in samples/ (text+pen, scanned, multi-page, hi-res)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    tmp = out / "_clean.pdf"
    build_clean_pdf(tmp, pages=3)
    # 1) multi-page scan-like PDF (raster pages with pen + uneven background)
    scan = pymupdf.open()
    for i in range(3):
        clean = render(tmp, i, 200)
        dirty, _ = add_pens(clean, "all", seed=i + 1)
        yy, xx = np.mgrid[0:dirty.shape[0], 0:dirty.shape[1]].astype(np.float32)
        shade = 0.93 + 0.07 * np.sin(xx / 400.0) * np.cos(yy / 500.0)
        dirty = np.clip(dirty.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", dirty, [cv2.IMWRITE_JPEG_QUALITY, 88])   # realistic scan
        pg = scan.new_page(width=PAGE_W, height=PAGE_H)
        pg.insert_image(pg.rect, stream=buf.tobytes())
    scan.save(str(out / "sample_scanned_pen.pdf")); scan.close()
    made.append(out / "sample_scanned_pen.pdf")
    # 2) text PDF: pen as *raster overlay* on a page that keeps real text -> flatten page
    doc = pymupdf.open(str(tmp))
    for i in range(1):
        clean = render(tmp, i, 200)
        dirty, _ = add_pens(clean, "all", seed=11)
        ok, buf = cv2.imencode(".jpg", dirty, [cv2.IMWRITE_JPEG_QUALITY, 90])
        doc[i].insert_image(doc[i].rect, stream=buf.tobytes(), overlay=True)
    doc.save(str(out / "sample_text_with_pen_overlay.pdf")); doc.close()
    made.append(out / "sample_text_with_pen_overlay.pdf")
    tmp.unlink()
    return made


if __name__ == "__main__":
    for p in build_samples(Path(__file__).resolve().parent.parent / "samples"):
        print("created", p)
