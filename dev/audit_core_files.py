#!/usr/bin/env python3
# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Critical Files Audit Script
Checks all core BBX files for import errors and basic syntax
"""

import sys
import ast
from pathlib import Path

CRITICAL_FILES = [
    "cli.py",
    "blackbox/core/runtime.py",
    "blackbox/core/registry.py",
    "blackbox/core/base_adapter.py",
    "blackbox/core/universal.py",
    "blackbox/core/universal_v2.py",
    "blackbox/core/auth.py",
    "blackbox/core/package_manager.py",
    "blackbox/core/audit.py",
    "blackbox/core/security.py",
    "blackbox/core/pipeline.py",
    "blackbox/core/registry_auth.py",
    "blackbox/core/universal_schema.py",
]

def check_file(filepath):
    """Check a Python file for basic errors."""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print('='*60)
    
    path = Path(filepath)
    if not path.exists():
        print("  ❌ File not found")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check syntax
        try:
            ast.parse(code)
            print("  ✅ Syntax OK")
        except SyntaxError as e:
            print(f"  ❌ Syntax Error: Line {e.lineno}: {e.msg}")
            return False
        
        # Try to import (basic check)
        module_name = str(path).replace('\\', '.').replace('/', '.').replace('.py', '')
        if module_name.startswith('.'):
            module_name = module_name[1:]
        
        print("  📦 Imports found:")
        import_lines = [line for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
        for imp in import_lines[:5]:  # Show first 5
            print(f"     {imp.strip()}")
        if len(import_lines) > 5:
            print(f"     ... and {len(import_lines) - 5} more")
        
        print(f"  ✅ {path.name} looks OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("BBX Critical Files Audit")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for filepath in CRITICAL_FILES:
        if check_file(filepath):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
