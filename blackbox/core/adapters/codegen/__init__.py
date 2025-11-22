# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Code Generation Adapters Package

Provides adapters for code generation, templating, and scaffolding.
"""

from blackbox.core.adapters.codegen.fs import FileSystemGenAdapter
from blackbox.core.adapters.codegen.template import TemplateAdapter

__all__ = ["TemplateAdapter", "FileSystemGenAdapter"]
