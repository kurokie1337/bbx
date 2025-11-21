import subprocess
import sys
import time
import webbrowser
import os
import requests
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = ['fastapi', 'uvicorn', 'requests', 'yaml']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def wait_for_server(url, timeout=10):
    """Wait for server to be responsive."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False

def main():
    print("\n" + "="*60)
    print("🚀 Starting Blackbox Workflow Engine")
    print("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Get paths
    root_dir = Path(__file__).parent.absolute()
    api_script = root_dir / "api_server.py"
    
    if not api_script.exists():
        print(f"❌ Could not find api_server.py at {api_script}")
        sys.exit(1)
        
    # Start API Server
    print("\n📡 Starting API Server...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    
    process = subprocess.Popen(
        [sys.executable, str(api_script)],
        cwd=str(root_dir),
        env=env
    )
    
    api_url = "http://localhost:8000"
    
    try:
        # Wait for server
        if wait_for_server(api_url):
            print(f"✅ Server is ready at {api_url}")
            print("🌐 Opening Visual Editor...")
            
            # Open browser
            webbrowser.open(api_url)
            
            print("\nPress Ctrl+C to stop the server")
            process.wait()
        else:
            print("❌ Server failed to start within timeout")
            process.terminate()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        process.terminate()
        print("✅ Done")

if __name__ == "__main__":
    main()
