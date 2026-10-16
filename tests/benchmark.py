"""Quality benchmark on synthetic pages with known ground truth.
Measures (1) pen removal completeness, (2) damage to original detail, (3) PDF integrity."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pymupdf

from config import Settings
from core import color_detector, pdf_exporter, processing_pipeline as pp
from tests import synthetic as syn


def run():
    tmp = Path(tempfile.mkdtemp())
    pdf = syn.build_clean_pdf(tmp / "clean.pdf")
    clean = syn.render(pdf, 0, 200)
    print(f"{'case':<26}{'mode':<14}{'coverage':>9}{'residue':>9}{'far-changed px':>16}{'text kept':>10}{'pen-px err':>11}")
    for case in ("red", "blue", "light", "all"):
        dirty, gt = syn.add_pens(clean, case)
        g = gt > 0
        far = cv2.dilate(gt, np.ones((31, 31), np.uint8)) == 0
        dark = (clean.max(axis=2) < 90) & far
        for mode in ("conservative", "balanced", "advanced"):
            r = pp.process_page(dirty, Settings(mode=mode))
            high, _ = color_detector.detect_pen_pixels(r.restored, Settings(mode=mode))
            near = cv2.dilate(gt, np.ones((9, 9), np.uint8)) > 0
            residue = ((high > 0) & near).sum() / g.sum()
            cov = (r.final_mask[g] > 0).mean()
            d = np.abs(r.restored.astype(int) - clean.astype(int)).max(axis=2)
            changed = int((np.abs(r.restored.astype(int) - dirty.astype(int)).max(axis=2)[far] > 0).sum())
            print(f"{case:<26}{mode:<14}{cov:>9.3f}{residue:>9.4f}{changed:>16d}{(d[dark] < 40).mean():>10.4f}{d[g].mean():>11.1f}")
    # export integrity
    dirty, _ = syn.add_pens(clean, "all")
    res = pp.process_page(dirty, Settings())
    png = tmp / "p0.png"
    cv2.imwrite(str(png), res.restored)
    rep = pdf_exporter.export_pdf(pdf, tmp / "o.pdf", {0: str(png)}, Settings())
    d = pymupdf.open(str(tmp / "o.pdf"))
    print(f"\nexport verified={rep.verified} pages={d.page_count} searchable_text_on_processed_page={bool(d[0].search_for('Annual Report'))}")


if __name__ == "__main__":
    run()
