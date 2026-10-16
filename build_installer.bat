@echo off
REM One-click: build .exe with PyInstaller, then package Setup.exe with Inno Setup 6.
REM Requires: Python 3.12 on PATH, Inno Setup 6 (https://jrsoftware.org/isdl.php)
setlocal
cd /d "%~dp0"

if not exist venv ( python -m venv venv || goto :err )
call venv\Scripts\activate || goto :err
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err

pyinstaller --noconfirm --clean --windowed --name LocalPDFPenRemover ^
  --collect-submodules PySide6 --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.QtPdf --exclude-module PySide6.QtQuick --exclude-module PySide6.QtQml ^
  main.py || goto :err

set ISCC=
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe
if "%ISCC%"=="" (
  echo [!] Inno Setup 6 not found - only the .exe folder was built: dist\LocalPDFPenRemover\
  goto :end
)
"%ISCC%" installer.iss || goto :err
echo.
echo DONE: installer_output\LocalPDFPenRemover_Setup_1.0.0.exe
goto :end
:err
echo BUILD FAILED
exit /b 1
:end
endlocal
