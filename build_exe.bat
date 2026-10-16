@echo off
REM Build a Windows executable (run inside the activated virtual environment)
pip install -r requirements.txt
pyinstaller --noconfirm --clean --windowed --name LocalPDFPenRemover ^
  --collect-submodules PySide6 --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.QtPdf --exclude-module PySide6.QtQuick --exclude-module PySide6.QtQml ^
  main.py
echo.
echo Output: dist\LocalPDFPenRemover\LocalPDFPenRemover.exe
