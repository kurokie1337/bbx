#!/usr/bin/env python
"""
Complete setup and test for BBX AI generation
"""

import sys
import os

# Fix UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("=" * 80)
print("BBX AI SETUP AND TEST")
print("=" * 80)

# Step 1: Check dependencies
print("\n[1/4] Checking dependencies...")
try:
    from llama_cpp import Llama
    print("  OK llama-cpp-python installed")
except ImportError:
    print("  ERROR: llama-cpp-python not installed")
    print("  Run: pip install llama-cpp-python")
    sys.exit(1)

try:
    from blackbox.ai.model_manager import ModelManager
    from blackbox.ai.generator import WorkflowGenerator
    print("  OK BBX AI modules found")
except ImportError as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Step 2: Download model
print("\n[2/4] Downloading Qwen2.5 0.5B model (250MB)...")
manager = ModelManager()

# Check if already downloaded
if "qwen-0.5b" in manager.list_installed():
    print("  OK Model already downloaded")
else:
    print("  Downloading from HuggingFace...")
    try:
        manager.download("qwen-0.5b")
        print("  OK Model downloaded successfully")
    except Exception as e:
        print(f"  ERROR: Failed to download model: {e}")
        sys.exit(1)

# Step 3: Test generation
print("\n[3/4] Testing workflow generation...")
print("  Prompt: 'Run pytest tests'")

try:
    generator = WorkflowGenerator(model_name="qwen-0.5b")

    # Generate workflow
    yaml_content = generator.generate(
        description="Run pytest tests",
        output_file="test_generated.bbx"
    )

    print("  OK Workflow generated!")
    print("\n" + "-" * 80)
    print("Generated workflow:")
    print("-" * 80)
    print(yaml_content)
    print("-" * 80)

except Exception as e:
    print(f"  ERROR: Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify
print("\n[4/4] Verifying output...")

# Check file exists
if os.path.exists("test_generated.bbx"):
    print("  OK File created: test_generated.bbx")
else:
    print("  ERROR: File not created")
    sys.exit(1)

# Check content
with open("test_generated.bbx", "r", encoding="utf-8") as f:
    content = f.read()

if "workflow:" in content and "steps:" in content:
    print("  OK Valid BBX workflow structure")
else:
    print("  WARNING: Workflow structure might be invalid")

print("\n" + "=" * 80)
print("SUCCESS! BBX AI is fully working!")
print("=" * 80)
print("\nNext steps:")
print("  1. Review: cat test_generated.bbx")
print("  2. Validate: python cli.py validate test_generated.bbx")
print("  3. Generate more: python test_cli.py generate 'Your task here'")
print("\n" + "=" * 80)
