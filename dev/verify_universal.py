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

import asyncio
import yaml
from blackbox.core.universal import UniversalAdapter

async def test_universal():
    print("Testing Universal Adapter...")
    
    # 1. Load Definition
    with open("blackbox/library/aws_cli.yaml", "r") as f:
        aws_def = yaml.safe_load(f)
        
    # 2. Initialize Adapter
    adapter = UniversalAdapter(aws_def)
    
    # 3. Mock Inputs
    inputs = {
        "service": "s3",
        "action": "ls", # s3 ls is simpler than list-buckets for testing
        "args": {}
    }
    
    # 4. Execute (Dry Run / Mock)
    # Since we might not have credentials, we expect a specific error or we mock run_command
    # For this test, let's just see if it renders the command correctly.
    
    print(f"Definition: {aws_def}")
    
    # We can inspect the internal rendering if we subclass or mock
    # But let's try to run it. If docker is missing or creds missing, it will fail, 
    # but we'll see the attempt.
    
    try:
        result = await adapter.execute("run", inputs)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Execution failed (expected if no creds): {e}")

if __name__ == "__main__":
    asyncio.run(test_universal())
