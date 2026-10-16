"""PDF export.

* Pages that were NOT processed are copied from the source untouched
  (text, links, vector objects, annotations stay intact).
* Processed pages are rebuilt from the cleaned raster at full resolution
  (PNG lossless by default).  Because the page becomes an image, an invisible
  text layer is re-added from the original so text stays searchable/selectable
  (only for characters Latin-1 can encode -- see ExportReport.warnings).
* Source file is never overwritten; the result is re-opened and verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import pymupdf

from config import Settings


class ExportError(Exception):
    pass


@dataclass
class ExportReport:
    output: str
    pages_total: int
    pages_replaced: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    verified: bool = False


def default_output_name(src: str | Path, folder: str | Path | None = None) -> Path:
    src = Path(src)
    return Path(folder or src.parent) / f"{src.stem}_cleaned.pdf"


def _add_text_layer(new_page, src_page, warnings: list, page_no: int) -> None:
    lost = 0
    for wx0, wy0, wx1, wy1, word, *_ in src_page.get_text("words"):
        r = pymupdf.Rect(wx0, wy0, wx1, wy1) * src_page.rotation_matrix   # into visible orientation
        r.normalize()
        x0, y0, x1, y1 = r
        h = max(1.0, min(y1 - y0, x1 - x0) if src_page.rotation in (90, 270) else y1 - y0)
        try:
            word.encode("latin-1")
        except UnicodeEncodeError:
            lost += 1
            continue
        try:
            new_page.insert_text((x0, y1 - 0.2 * h), word, fontsize=h * 0.8,
                                 fontname="helv", render_mode=3)
        except Exception:                                       # noqa: BLE001
            lost += 1
    if lost:
        warnings.append(f"หน้า {page_no + 1}: ข้อความ {lost} คำ (อักขระนอก Latin-1 เช่น ภาษาไทย) "
                        "ไม่ถูกเก็บเป็น Text Layer ที่ค้นหาได้")


def export_pdf(src_path: str | Path, out_path: str | Path, page_images: dict[int, str],
               settings: Settings | None = None, password: str | None = None,
               progress: Callable[[int, int], None] | None = None,
               cancel: Callable[[], bool] | None = None) -> ExportReport:
    """page_images: {page_index: path of cleaned PNG}."""
    settings = settings or Settings()
    src_path, out_path = Path(src_path), Path(out_path)
    if out_path.resolve() == src_path.resolve():
        raise ExportError("ห้ามเขียนทับไฟล์ต้นฉบับ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = pymupdf.open(str(src_path))
    if src.needs_pass and not src.authenticate(password or ""):
        raise ExportError("ไฟล์ต้นฉบับมีรหัสผ่าน")
    out = pymupdf.open()
    report = ExportReport(str(out_path), src.page_count)
    try:
        for i in range(src.page_count):
            if cancel and cancel():
                raise ExportError("ยกเลิกการส่งออก")
            if i in page_images:
                sp = src[i]
                rect = sp.rect
                np_ = out.new_page(width=rect.width, height=rect.height)
                if settings.image_format == "jpeg":
                    img = cv2.imread(page_images[i])
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(settings.jpeg_quality)])
                    np_.insert_image(np_.rect, stream=buf.tobytes())
                else:
                    np_.insert_image(np_.rect, filename=page_images[i])
                had_text = sp.get_text("text").strip() != ""
                if settings.keep_text_layer and had_text:
                    _add_text_layer(np_, sp, report.warnings, i)
                elif had_text:
                    report.warnings.append(f"หน้า {i + 1}: ข้อความถูกแปลงเป็นภาพ คัดลอก/ค้นหาไม่ได้")
                if sp.get_links():
                    for lk in sp.get_links():
                        try:
                            lk = dict(lk)
                            if lk.get("kind") == pymupdf.LINK_GOTO:
                                lk.pop("xref", None)
                            np_.insert_link(lk)
                        except Exception:                       # noqa: BLE001
                            pass
                if sp.rotation:
                    report.warnings.append(f"หน้า {i + 1}: การหมุนถูกฝังในภาพ (ทิศทางที่เห็นเหมือนเดิม)")
                report.pages_replaced.append(i)
            else:
                out.insert_pdf(src, from_page=i, to_page=i)
            if progress:
                progress(i + 1, src.page_count)
        try:
            out.set_metadata(src.metadata or {})
            toc = [t for t in src.get_toc() if 1 <= t[2] <= src.page_count]
            if toc:
                out.set_toc(toc)
        except Exception:                                       # noqa: BLE001
            report.warnings.append("คัดลอก Metadata/Bookmark ไม่สมบูรณ์")
        out.save(str(out_path), garbage=3, deflate=True)
    finally:
        out.close()
        src.close()
    _verify(src_path, out_path, report, password)
    return report


def _verify(src_path: Path, out_path: Path, report: ExportReport, password: str | None):
    try:
        a = pymupdf.open(str(src_path))
        if a.needs_pass:
            a.authenticate(password or "")
        b = pymupdf.open(str(out_path))
        try:
            if a.page_count != b.page_count:
                raise ExportError(f"จำนวนหน้าไม่ตรงกัน ({a.page_count} != {b.page_count})")
            for i in range(a.page_count):
                ra, rb = a[i].rect, b[i].rect
                if abs(ra.width - rb.width) > 0.5 or abs(ra.height - rb.height) > 0.5:
                    raise ExportError(f"ขนาดหน้า {i + 1} ไม่ตรงกับต้นฉบับ")
            b[0].get_pixmap(matrix=pymupdf.Matrix(0.2, 0.2))     # must be renderable
            report.verified = True
        finally:
            a.close()
            b.close()
    except ExportError:
        raise
    except Exception as e:                                       # noqa: BLE001
        raise ExportError(f"ไฟล์ผลลัพธ์ตรวจสอบไม่ผ่าน: {e}") from e
