"""Headless GUI smoke test (offscreen Qt)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QMessageBox

from tests import synthetic as syn


def wait_job(app, win, timeout=90):
    t = time.time()
    while win.worker is not None and win.worker.isRunning() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()


def test_gui_flow(tmp_path, monkeypatch):
    for n in ("information", "critical", "warning"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    src = [p for p in syn.build_samples(tmp_path) if "scanned" in p.name][0]
    win = MainWindow()
    win.show()
    win.open_pdf(str(src))
    assert win.loader.page_count == 3 and win.thumbs.count() == 3 and win.orig_img is not None

    win.run_preview(); wait_job(app, win)
    assert win.pages[0].auto_mask is not None and win.pages[0].auto_mask.any()

    win.preview.left.set_tool("add")                        # brush, undo/redo
    win.on_brush_start(); win.on_brush_move(QPointF(10, 10), QPointF(80, 10)); win.on_brush_end()
    assert win.pages[0].editor.has_edits()
    win.undo(); assert not win.pages[0].editor.has_edits()
    win.redo(); assert win.pages[0].editor.has_edits()

    win.run_process([0]); wait_job(app, win)
    assert win.pages[0].processed_path and win.processed_img is not None
    for mode in ("original", "processed", "split", "slider"):  # all comparison modes
        win.preview.mode.setCurrentIndex([k for _, k in win.preview.VIEW_MODES].index(mode))
    win.goto(1); win.run_process_all(); wait_job(app, win)
    assert all(s.processed_path for s in win.pages.values())

    win.settings = lambda: win.panel.settings()
    from core import pdf_exporter
    imgs = {i: s.processed_path for i, s in win.pages.items()}
    rep = pdf_exporter.export_pdf(src, tmp_path / "gui_out.pdf", imgs, win.panel.settings())
    assert rep.verified
    win.close()


def test_cancel(tmp_path, monkeypatch):
    from gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    src = [p for p in syn.build_samples(tmp_path) if "scanned" in p.name][0]
    win = MainWindow()
    win.open_pdf(str(src))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    win.run_process_all(); win.cancel_job(); wait_job(app, win)
    assert not win.worker.isRunning()
    win.close()
