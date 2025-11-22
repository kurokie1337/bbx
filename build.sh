#!/bin/bash
# BBX Build Script for Linux/macOS
# Builds single-file executable using PyInstaller

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         BBX Binary Build Script (Linux/macOS)             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "[ERROR] PyInstaller not found. Installing..."
    pip3 install pyinstaller
fi

echo "[1/4] Cleaning previous builds..."
rm -rf build/ dist/ bbx

echo "[2/4] Building BBX binary with PyInstaller..."
echo ""
pyinstaller --clean --noconfirm bbx.spec

if [ ! -f "dist/bbx" ]; then
    echo ""
    echo "[ERROR] Build failed! Binary not found in dist/"
    exit 1
fi

echo ""
echo "[3/4] Moving binary to root..."
mv dist/bbx bbx
chmod +x bbx
echo "✅ Binary created: bbx"

echo ""
echo "[4/4] Testing binary..."
./bbx --version || echo "[WARNING] Binary test failed, but file was created"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    BUILD COMPLETE                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Binary location: $(pwd)/bbx"
echo ""
echo "You can now run: ./bbx --help"
echo ""
