import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from blackbox.ai.generator import WorkflowGenerator
except ImportError:
    print("Failed to import WorkflowGenerator. Run verify_system.py first.")
    sys.exit(1)

def generate_demo():
    print("=== Generating Crazy App Demo ===")
    
    # Initialize generator (will try to download model if missing)
    try:
        generator = WorkflowGenerator()
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return

    prompt = "Create a crazy application that prints infinite recursive ASCII art logs to the console using a python script, demonstrating a loop that never ends but looks cool."
    output_file = "examples/crazy_app.bbx"
    
    print(f"Prompt: {prompt}")
    
    try:
        yaml_content = generator.generate(prompt, output_file=output_file)
        print("\n=== Generated Workflow Content ===")
        print(yaml_content)
        print("==================================")
        print(f"✅ Workflow saved to {output_file}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    generate_demo()
