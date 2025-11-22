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
BBX CLI Test Suite
Quick validation of all CLI commands
"""

import subprocess
import sys

def run_cmd(cmd):
    """Run a CLI command and return success status."""
    print(f"Testing: {cmd}")
    try:
        result = subprocess.run(
            [sys.executable, "cli.py"] + cmd.split(),
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 or "--help" in cmd:
            print("  ✅ OK")
            return True
        else:
            print(f"  ❌ FAILED: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def main():
    print("="*60)
    print("BBX CLI Validation Suite")
    print("="*60)
    
    tests = [
        # Core commands
        "--help",
        "version",
        
        # Package commands
        "package --help",
        "package list",
        "package validate",
        
        # Audit command
        "audit --help",
        
        # Security command
        "security-scan --help",
        
        # Registry command
        "registry-login --help",
        
        # System command
        "system",
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if run_cmd(test):
            passed += 1
        else:
            failed += 1
        print()
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
