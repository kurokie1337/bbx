# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Dev workflows for BBX self-hosting
# BBX develops itself using BBX!

This directory contains BBX workflows for developing BBX itself.

## Workflows

- **`build.bbx`** - Build BBX from source
- **`test.bbx`** - Run all tests with coverage
- **`lint.bbx`** - Lint BBX codebase
- **`release.bbx`** - Create new BBX release

## Usage

```bash
# Build BBX using BBX
bbx run dev/build.bbx

# Test BBX using BBX
bbx run dev/test.bbx

# Create release
bbx run dev/release.bbx
```

## Self-Hosting Philosophy

BBX is developed USING BBX. This proves:
- BBX is production-ready
- BBX can handle complex workflows
- "Dog fooding" ensures quality
- Single paradigm for everything

If BBX can build itself, it can build anything! 🚀
