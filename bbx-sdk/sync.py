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

import yaml
import logging
from typing import List, Optional, Dict
from pathlib import Path

from .client import BlackboxClient
from .models import WorkflowCreate, WorkflowUpdate, WorkflowResponse

logger = logging.getLogger(__name__)

class WorkflowSyncer:
    def __init__(self, client: BlackboxClient):
        self.client = client

    def load_local_workflow(self, file_path: str) -> Dict:
        """Load and parse a local BBX file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
                data = yaml.safe_load(content)
                return {"content": content, "data": data, "filename": path.name}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {file_path}: {e}")

    def sync_file_to_blackbox(self, file_path: str, update_existing: bool = True) -> Optional[WorkflowResponse]:
        """Sync a local file to BLACKBOX."""
        logger.info(f"Syncing file: {file_path}")
        
        try:
            local_data = self.load_local_workflow(file_path)
            bbx_data = local_data["data"]
            
            # Extract workflow metadata
            if "workflow" not in bbx_data:
                raise ValueError("Invalid BBX format: missing 'workflow' root key")
                
            wf_meta = bbx_data["workflow"]
            wf_id = wf_meta.get("id")
            wf_name = wf_meta.get("name", local_data["filename"])
            wf_desc = wf_meta.get("description")
            
            # Check if workflow exists
            existing_workflow = None
            if update_existing and wf_id:
                try:
                    existing_workflow = self.client.get_workflow(wf_id)
                except Exception:
                    logger.info(f"Workflow {wf_id} not found, will create new")

            if existing_workflow:
                # Update
                logger.info(f"Updating existing workflow: {wf_id}")
                update = WorkflowUpdate(
                    name=wf_name,
                    description=wf_desc,
                    bbx_yaml=local_data["content"]
                )
                return self.client.update_workflow(wf_id, update)
            else:
                # Create
                logger.info(f"Creating new workflow: {wf_name}")
                create = WorkflowCreate(
                    name=wf_name,
                    description=wf_desc,
                    bbx_yaml=local_data["content"]
                )
                return self.client.create_workflow(create)
                
        except Exception as e:
            logger.error(f"Failed to sync {file_path}: {e}")
            raise

    def sync_directory(self, directory: str) -> List[WorkflowResponse]:
        """Sync all .bbx files in a directory."""
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
            
        results = []
        for file_path in path.glob("*.bbx"):
            try:
                result = self.sync_file_to_blackbox(str(file_path))
                results.append(result)
            except Exception:
                logger.error(f"Skipping {file_path} due to error")
                
        return results

    def download_workflow(self, workflow_id: str, output_dir: str) -> str:
        """Download a workflow from BLACKBOX to a local file."""
        workflow = self.client.get_workflow(workflow_id)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{workflow.id}.bbx"
        # Try to use name for filename if it's safe
        safe_name = "".join([c for c in workflow.name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        if safe_name:
            filename = f"{safe_name}.bbx"
            
        file_path = output_path / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(workflow.bbx_yaml)
            
        logger.info(f"Downloaded workflow {workflow_id} to {file_path}")
        return str(file_path)
