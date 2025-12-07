import sys
import os
from pathlib import Path

def check_llama_cpp():
    print("Checking llama-cpp-python...")
    try:
        import llama_cpp
        print("✅ llama-cpp-python is installed.")
        return True
    except ImportError:
        print("❌ llama-cpp-python is NOT installed.")
        return False

def check_models():
    print("\nChecking for LLM models...")
    models_dir = Path.home() / ".bbx" / "models"
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # We look for qwen-0.5b which seems to be the default
    found = list(models_dir.glob("*.gguf"))
    if found:
        print(f"✅ Found models: {[f.name for f in found]}")
        return True
    else:
        print("❌ No .gguf models found in models directory.")
        return False

def check_workflow_generator():
    print("\nChecking WorkflowGenerator import...")
    try:
        # Add current directory to path so we can import blackbox
        sys.path.append(os.getcwd())
        from blackbox.ai.generator import WorkflowGenerator
        print("✅ WorkflowGenerator imported successfully.")
        return True
    except ImportError as e:
        print(f"❌ Failed to import WorkflowGenerator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during import: {e}")
        return False

if __name__ == "__main__":
    print("=== BBX System Verification ===")
    
    llama_ok = check_llama_cpp()
    models_ok = check_models()
    gen_ok = check_workflow_generator()
    
    if llama_ok and models_ok and gen_ok:
        print("\n✅ SYSTEM READY FOR GENERATION")
        sys.exit(0)
    else:
        print("\n⚠️ SYSTEM ISSUES DETECTED")
        if not llama_ok:
            print("  - Install llama-cpp: pip install llama-cpp-python")
        if not models_ok:
            print("  - Models missing. They might be downloaded automatically.")
        sys.exit(1)
