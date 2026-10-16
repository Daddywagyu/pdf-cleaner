"""Local PDF Pen Remover - entry point (100% offline)."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LocalPDFPenRemover"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "app.log"
    h = RotatingFileHandler(path, maxBytes=512_000, backupCount=2, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h])
    return path


def main() -> int:
    setup_logging()
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".pdf"):
        win.open_pdf(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
