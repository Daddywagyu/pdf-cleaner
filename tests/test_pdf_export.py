import hashlib
from pathlib import Path

import cv2
import pymupdf
import pytest

from config import Settings
from core import pdf_exporter, processing_pipeline as pp, color_detector
from core.pdf_loader import PdfLoader, PdfOpenError, PdfPasswordRequired, safe_dpi
from tests import synthetic as syn


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean_png(tmp, pdf, i=0, dpi=200):
    p = str(tmp / f"p{i}.png")
    cv2.imwrite(p, syn.render(pdf, i, dpi))
    return p


def test_export_roundtrip_keeps_other_pages_and_original(tmp_path, clean_pdf):
    before = sha(clean_pdf)
    out = tmp_path / "x_cleaned.pdf"
    rep = pdf_exporter.export_pdf(clean_pdf, out, {1: clean_png(tmp_path, clean_pdf, 1)}, Settings())
    assert rep.verified and rep.pages_replaced == [1]
    d = pymupdf.open(str(out))
    assert d.page_count == 3
    assert d[0].search_for("Annual Report")             # untouched page keeps real text
    assert d[1].search_for("Annual Report")             # text layer re-added on processed page
    assert d[1].get_images()                            # processed page is an image
    assert sha(clean_pdf) == before                     # original never modified


def test_refuses_to_overwrite_source(tmp_path, clean_pdf):
    with pytest.raises(pdf_exporter.ExportError):
        pdf_exporter.export_pdf(clean_pdf, clean_pdf, {}, Settings())


def test_page_size_rotation_and_dpi(tmp_path):
    src = syn.build_clean_pdf(tmp_path / "rot.pdf", rotate=90)
    for dpi in (200, 300, 400):
        img = clean_png(tmp_path, src, 0, dpi)
        out = tmp_path / f"o{dpi}.pdf"
        pdf_exporter.export_pdf(src, out, {0: img}, Settings())
        d = pymupdf.open(str(out))
        assert abs(d[0].rect.width - 842) < 0.5 and abs(d[0].rect.height - 595) < 0.5
        w = d[0].get_images(full=True)[0][2]
        assert abs(w - 842 / 72 * dpi) < 3                  # full resolution is embedded


def test_no_text_layer_option_warns(tmp_path, clean_pdf):
    rep = pdf_exporter.export_pdf(clean_pdf, tmp_path / "a.pdf", {0: clean_png(tmp_path, clean_pdf)},
                                  Settings(keep_text_layer=False))
    assert any("ภาพ" in w for w in rep.warnings)
    assert not pymupdf.open(str(tmp_path / "a.pdf"))[0].search_for("Annual Report")


def test_toc_and_metadata_preserved(tmp_path, clean_pdf):
    d = pymupdf.open(str(clean_pdf))
    d.set_toc([[1, "Intro", 1], [1, "Second", 2]])
    d.set_metadata({"title": "T"})
    src = tmp_path / "toc.pdf"
    d.save(str(src))
    out = tmp_path / "toc_c.pdf"
    pdf_exporter.export_pdf(src, out, {0: clean_png(tmp_path, src)}, Settings())
    o = pymupdf.open(str(out))
    assert o.get_toc()[1][1] == "Second" and o.metadata["title"] == "T"


def test_loader_errors(tmp_path):
    with pytest.raises(PdfOpenError):
        PdfLoader(tmp_path / "missing.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not a pdf")
    with pytest.raises(PdfOpenError):
        PdfLoader(bad)
    d = pymupdf.open()
    d.new_page()
    enc = tmp_path / "enc.pdf"
    d.save(str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(PdfPasswordRequired):
        PdfLoader(enc)
    with pytest.raises(PdfPasswordRequired):
        PdfLoader(enc, "wrong")
    assert PdfLoader(enc, "secret").page_count == 1


def test_loader_info_and_memory_guard(clean_pdf):
    with PdfLoader(clean_pdf) as ld:
        assert ld.page_count == 3 and ld.page_size(0) == (595.0, 842.0) and ld.has_text(0)
        img, used = ld.render(0, 200)
        assert used == 200 and img.shape == (2339, 1653, 3)
        assert ld.thumbnail(0, 100).shape[0] == 100
    assert safe_dpi(595, 842, 400, max_pixels=5_000_000) < 400


def test_end_to_end_scanned_multipage(tmp_path):
    """Sample scan -> process all pages -> export -> re-open -> measure."""
    samples = syn.build_samples(tmp_path / "samples")
    src = [p for p in samples if "scanned" in p.name][0]
    s = Settings()
    imgs = {}
    with PdfLoader(src) as ld:
        for i in range(ld.page_count):
            img, _ = ld.render(i, 200)
            res = pp.process_page(img, s)
            p = str(tmp_path / f"r{i}.png")
            cv2.imwrite(p, res.restored)
            imgs[i] = p
            assert (res.final_mask > 0).sum() > 1000
    out = tmp_path / "scan_cleaned.pdf"
    rep = pdf_exporter.export_pdf(src, out, imgs, s)
    assert rep.verified
    with PdfLoader(out) as ld:
        assert ld.page_count == 3
        for i in range(3):
            img, _ = ld.render(i, 200)
            left = color_detector.detect_mask(img, s)
            assert int((left > 0).sum()) < 200, f"pen residue on page {i + 1}"
