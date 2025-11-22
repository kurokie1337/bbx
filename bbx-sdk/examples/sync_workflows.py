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
Workflow Synchronization Example

This example demonstrates:
- Syncing local .bbx files to BBX server
- Syncing entire directories
- Downloading workflows from server
"""

import sys
from pathlib import Path

# Add parent directory to path to import bbx_sdk
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbx_sdk import BlackboxClient, WorkflowSyncer

def main():
    # Initialize client
    print("Initializing BBX client...")
    client = BlackboxClient("http://localhost:8000")

    # Authenticate
    print("Authenticating...")
    try:
        client.authenticate("admin", "admin")
        print("✓ Authentication successful\n")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Initialize syncer
    syncer = WorkflowSyncer(client)

    # Example 1: Sync a single file
    print("Example 1: Syncing single file")
    print("-" * 50)

    # First, create a sample workflow file
    sample_file = Path("/tmp/sample_workflow.bbx")
    sample_file.write_text("""
workflow:
  id: sync_example
  name: Sync Example Workflow
  version: "6.0"
  description: Example workflow for sync demonstration

  steps:
    - id: log_message
      mcp: bbx.logger
      method: info
      inputs:
        message: "This workflow was synced via SDK!"
""")

    try:
        result = syncer.sync_file_to_blackbox(str(sample_file))
        print(f"✓ Synced workflow: {result.name}")
        print(f"  ID: {result.id}")
        print(f"  Status: {result.status}")
    except Exception as e:
        print(f"✗ Failed to sync file: {e}")

    # Example 2: Sync directory
    print("\nExample 2: Syncing directory")
    print("-" * 50)

    # Create sample directory with multiple workflows
    workflows_dir = Path("/tmp/workflows")
    workflows_dir.mkdir(exist_ok=True)

    for i in range(3):
        wf_file = workflows_dir / f"workflow_{i}.bbx"
        wf_file.write_text(f"""
workflow:
  id: batch_sync_{i}
  name: Batch Sync Workflow {i}
  version: "6.0"

  steps:
    - id: step_{i}
      mcp: bbx.logger
      method: info
      inputs:
        message: "Workflow {i} from batch sync!"
""")

    try:
        results = syncer.sync_directory(str(workflows_dir))
        print(f"✓ Synced {len(results)} workflows:")
        for wf in results:
            print(f"  - {wf.name} ({wf.id})")
    except Exception as e:
        print(f"✗ Failed to sync directory: {e}")

    # Example 3: Download workflow
    print("\nExample 3: Downloading workflow from server")
    print("-" * 50)

    # Get first workflow from server
    try:
        workflows = client.list_workflows(limit=1)
        if workflows.items:
            first_workflow = workflows.items[0]

            download_dir = Path("/tmp/downloaded_workflows")
            download_dir.mkdir(exist_ok=True)

            downloaded_path = syncer.download_workflow(
                first_workflow.id,
                str(download_dir)
            )
            print(f"✓ Downloaded workflow to: {downloaded_path}")
            print(f"  Workflow: {first_workflow.name}")
            print(f"  ID: {first_workflow.id}")
        else:
            print("⚠ No workflows available to download")
    except Exception as e:
        print(f"✗ Failed to download workflow: {e}")

    # Example 4: Update existing workflow
    print("\nExample 4: Updating existing workflow")
    print("-" * 50)

    # Modify the sample file
    updated_content = """
workflow:
  id: sync_example
  name: Sync Example Workflow (Updated)
  version: "6.0"
  description: This workflow was updated via sync

  steps:
    - id: log_message
      mcp: bbx.logger
      method: info
      inputs:
        message: "This workflow was UPDATED via SDK!"

    - id: log_update
      mcp: bbx.logger
      method: info
      inputs:
        message: "Update successful!"
      depends_on: [log_message]
"""
    sample_file.write_text(updated_content)

    try:
        result = syncer.sync_file_to_blackbox(str(sample_file), update_existing=True)
        print(f"✓ Updated workflow: {result.name}")
        print(f"  ID: {result.id}")
        print(f"  Version: {result.version}")
    except Exception as e:
        print(f"✗ Failed to update workflow: {e}")

    print("\n✓ Sync examples completed!")

if __name__ == "__main__":
    main()
