@echo off
REM Build GenowDatasetCleaner.exe
REM Run this script from the project folder with your venv activated.
REM
REM Prerequisites (one-time setup):
REM   pip install pyinstaller

echo === GenowDatasetCleaner build ===

REM Install PyInstaller if missing
python -c "import PyInstaller" 2>nul || (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous build artifacts
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

REM Build
pyinstaller genowCleaner.spec

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo BUILD COMPLETE!
echo Executable folder: dist\GenowDatasetCleaner\
echo Run:               dist\GenowDatasetCleaner\GenowDatasetCleaner.exe
pause
