# BBX Distribution Guide

**Single-File Binary Distribution for BBX Workflow Engine**

BBX can now be distributed as a standalone executable that requires **no Python installation**. This makes it a true "Linux-style tool" that works anywhere.

---

## 📦 Available Binaries

### Windows
- **File**: `bbx.exe`
- **Size**: ~32MB
- **Requirements**: Windows 10/11, Docker (optional for workflow execution)

### Linux/macOS
- **File**: `bbx`
- **Size**: ~30-35MB
- **Requirements**: Linux x64 / macOS 10.14+, Docker (optional)

---

## 🚀 Quick Start

### Download & Run (No Installation)

**Windows:**
```cmd
# Download bbx.exe to your folder
# Run directly:
bbx.exe --help
bbx.exe run my-workflow.bbx
```

**Linux/macOS:**
```bash
# Download bbx binary
chmod +x bbx
./bbx --help
./bbx run my-workflow.bbx
```

### Add to PATH (Optional)

**Windows:**
```cmd
# Add current folder to PATH, or move bbx.exe to:
C:\Program Files\BBX\bbx.exe
```

**Linux/macOS:**
```bash
# Move to system bin
sudo mv bbx /usr/local/bin/bbx

# Now use globally:
bbx --help
```

---

## 🏗️ Building from Source

### Prerequisites
- Python 3.10+
- pip
- Git

### Build Steps

**1. Clone repository**
```bash
git clone https://github.com/yourusername/bbx.git
cd bbx
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install pyinstaller
```

**3. Build binary**

**Windows:**
```cmd
build.bat
```

**Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

The binary will be created in the project root:
- Windows: `bbx.exe`
- Linux/macOS: `bbx`

---

## 📋 Build Configuration

### PyInstaller Spec File

The build is configured in `bbx.spec`:

```python
# Key features:
- Single-file executable (--onefile)
- Includes Standard Library (library/ folder)
- Includes templates (templates/ folder)
- Optimized with UPX compression
- Console application (CLI)
```

### What's Included

The binary bundles:
- ✅ BBX core engine
- ✅ All Python dependencies
- ✅ Standard Library YAML recipes
- ✅ Template files
- ✅ Docker integration
- ✅ Universal Adapter

### What's NOT Included (External)

- ❌ Docker Desktop (must be installed separately)
- ❌ User workflows (.bbx files)
- ❌ Docker images (downloaded on-demand)

---

## 🔍 Binary Verification

### Check Version
```bash
bbx --version
# Output: bbx.exe, version 1.0.0
```

### List Commands
```bash
bbx --help
```

### Run a Test Workflow
```bash
bbx run examples/hello_world.bbx
```

---

## 📦 Distribution Methods

### Method 1: Direct Download (GitHub Releases)

Upload binaries to GitHub Releases:
```bash
# Tag release
git tag v1.0.0
git push origin v1.0.0

# Upload to GitHub:
- bbx.exe (Windows)
- bbx-linux (Linux)
- bbx-macos (macOS)
```

### Method 2: Package Managers

**Homebrew (macOS/Linux):**
```bash
# Future:
brew install bbx
```

**Chocolatey (Windows):**
```powershell
# Future:
choco install bbx
```

**Scoop (Windows):**
```powershell
# Future:
scoop install bbx
```

### Method 3: Docker Image

For users who prefer containers:
```bash
docker run -v $(pwd):/workspace bbxio/bbx run workflow.bbx
```

---

## 🛡️ Security

### Code Signing (Future)

For production releases:

**Windows:**
- Sign with Authenticode certificate
- Prevents SmartScreen warnings

**macOS:**
- Sign with Apple Developer ID
- Notarize for Gatekeeper

**Linux:**
- GPG signature for package verification

---

## 📊 Build Size Optimization

### Current Size: ~32MB

**What takes space:**
- Python runtime: ~15MB
- Dependencies (FastAPI, Pydantic, etc.): ~12MB
- BBX code: ~3MB
- Standard Library: ~100KB

### Optimization Tips:

1. **Exclude dev dependencies** (already done):
   - pytest, black, mypy, etc.

2. **UPX Compression** (enabled):
   - Reduces binary size by ~30%

3. **Strip debug symbols** (future):
   - Can reduce another ~5-10%

---

## 🔧 Troubleshooting

### "File is too large"
- Normal! Single-file binaries are larger than scripts
- Consider distributing as zip/tar.gz

### "Antivirus blocks bbx.exe"
- Common with PyInstaller binaries
- Sign the binary (Windows/macOS)
- Whitelist in antivirus

### "ModuleNotFoundError" when running
- Rebuild with: `pyinstaller --clean bbx.spec`
- Check hiddenimports in bbx.spec

### "Docker not found"
- Binary works, but needs Docker for workflows
- Install Docker Desktop separately

---

## 🌍 Cross-Platform Builds

### Build on Native Platform

**Build Windows binary:**
- Must build on Windows

**Build Linux binary:**
- Must build on Linux (or use Docker)

**Build macOS binary:**
- Must build on macOS

### Using GitHub Actions (Automated)

Create `.github/workflows/release.yml`:

```yaml
name: Release Binaries

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt pyinstaller
      - run: build.bat
      - uses: actions/upload-artifact@v3
        with:
          name: bbx-windows
          path: bbx.exe

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt pyinstaller
      - run: chmod +x build.sh && ./build.sh
      - uses: actions/upload-artifact@v3
        with:
          name: bbx-linux
          path: bbx
```

---

## 📈 Version Management

### Updating Binary Version

1. Update version in `cli.py`:
```python
@click.version_option(version="1.1.0")
```

2. Rebuild:
```bash
build.bat  # or build.sh
```

3. Tag release:
```bash
git tag v1.1.0
git push origin v1.1.0
```

---

## 📝 Changelog

### v1.0.0 (Initial Release)
- ✅ Single-file executable
- ✅ Includes Standard Library
- ✅ Universal Adapter support
- ✅ Cross-platform builds

### Future Versions
- [ ] Code signing (Windows/macOS)
- [ ] Auto-update mechanism
- [ ] Homebrew/Chocolatey packages
- [ ] Smaller binary size (~20MB goal)

---

## 🤝 Contributing

### Building for Development

For faster iteration during development:

```bash
# Don't use --onefile (faster builds)
pyinstaller cli.py

# Binary in dist/cli/cli.exe
```

---

## 📚 Resources

- **PyInstaller Docs**: https://pyinstaller.org/
- **GitHub Releases**: https://docs.github.com/en/repositories/releasing-projects
- **Code Signing**: https://learn.microsoft.com/en-us/windows/win32/seccrypto/

---

## ✅ Distribution Checklist

Before releasing a new version:

- [ ] Update version in `cli.py`
- [ ] Update CHANGELOG.md
- [ ] Test binary on clean machine (no Python)
- [ ] Build for all platforms (Windows/Linux/macOS)
- [ ] Sign binaries (Windows/macOS)
- [ ] Create GitHub Release
- [ ] Upload binaries to release
- [ ] Update download links in README.md

---

**BBX is now a true standalone tool.**

No Python, no pip, no dependencies. Just download and run.

*"Make infrastructure automation boring again."*
