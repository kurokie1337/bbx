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
Basic BBX SDK Usage Example

This example demonstrates:
- Authenticating with BBX API
- Creating a workflow
- Listing workflows
- Executing a workflow
- Checking execution status
"""

import sys
from pathlib import Path

# Add parent directory to path to import bbx_sdk
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbx_sdk import BlackboxClient, WorkflowCreate

def main():
    # Initialize client
    print("Initializing BBX client...")
    client = BlackboxClient("http://localhost:8000")

    # Authenticate
    print("Authenticating...")
    try:
        client.authenticate("admin", "admin")
        print("✓ Authentication successful")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Create a simple workflow
    print("\nCreating workflow...")
    workflow_yaml = """
workflow:
  id: hello_world_sdk
  name: Hello World from SDK
  version: "6.0"
  description: Simple workflow created via BBX Python SDK

  steps:
    - id: greet
      mcp: bbx.logger
      method: info
      inputs:
        message: "Hello from BBX Python SDK! 🚀"

    - id: timestamp
      mcp: bbx.system
      method: execute
      inputs:
        command: "date"
      depends_on: [greet]

    - id: summary
      mcp: bbx.logger
      method: info
      inputs:
        message: "Workflow completed successfully!"
      depends_on: [timestamp]
"""

    new_workflow = WorkflowCreate(
        name="Hello World from SDK",
        description="Simple workflow created via BBX Python SDK",
        bbx_yaml=workflow_yaml
    )

    try:
        created = client.create_workflow(new_workflow)
        print(f"✓ Workflow created: {created.id}")
        print(f"  Name: {created.name}")
        print(f"  Status: {created.status}")
        print(f"  Created: {created.created_at}")
    except Exception as e:
        print(f"✗ Failed to create workflow: {e}")
        return

    # List all workflows
    print("\nListing workflows...")
    try:
        workflows = client.list_workflows(limit=10)
        print(f"✓ Found {workflows.total} workflows:")
        for wf in workflows.items[:5]:  # Show first 5
            print(f"  - {wf.name} ({wf.status})")
    except Exception as e:
        print(f"✗ Failed to list workflows: {e}")

    # Execute the workflow
    print(f"\nExecuting workflow {created.id}...")
    try:
        execution = client.execute_workflow(created.id)
        print(f"✓ Execution started: {execution.id}")
        print(f"  Status: {execution.status}")
        print(f"  Started at: {execution.started_at}")
    except Exception as e:
        print(f"✗ Failed to execute workflow: {e}")
        return

    # Check execution status
    print("\nChecking execution status...")
    try:
        status = client.get_execution_status(execution.id)
        print(f"✓ Execution status: {status.status}")
        if status.current_step:
            print(f"  Current step: {status.current_step}")
        if status.outputs:
            print(f"  Outputs: {status.outputs}")
    except Exception as e:
        print(f"✗ Failed to get status: {e}")

    # Get workflow details
    print(f"\nGetting workflow details...")
    try:
        workflow = client.get_workflow(created.id)
        print(f"✓ Workflow details:")
        print(f"  ID: {workflow.id}")
        print(f"  Name: {workflow.name}")
        print(f"  Status: {workflow.status}")
        print(f"  Version: {workflow.version}")
    except Exception as e:
        print(f"✗ Failed to get workflow: {e}")

    print("\n✓ Example completed successfully!")

if __name__ == "__main__":
    main()
