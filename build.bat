@echo off
REM BBX Build Script for Windows
REM Builds single-file executable using PyInstaller

echo ╔═══════════════════════════════════════════════════════════╗
echo ║           BBX Binary Build Script (Windows)               ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller
        exit /b 1
    )
)

echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist bbx.exe del /q bbx.exe

echo [2/4] Building BBX binary with PyInstaller...
echo.
pyinstaller --clean --noconfirm bbx.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [3/4] Moving binary to root...
if exist dist\bbx.exe (
    move dist\bbx.exe bbx.exe
    echo ✅ Binary created: bbx.exe
) else (
    echo [ERROR] Binary not found in dist/
    exit /b 1
)

echo.
echo [4/4] Testing binary...
bbx.exe --version

if errorlevel 1 (
    echo [WARNING] Binary test failed, but file was created
) else (
    echo ✅ Binary test passed!
)

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║                    BUILD COMPLETE                         ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo Binary location: %CD%\bbx.exe
echo.
echo You can now run: bbx.exe --help
echo.
