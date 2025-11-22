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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from blackbox.core.runtime import run_file
from blackbox.core.events import EventBus
import os

app = FastAPI(title="Blackbox Server")
event_bus = EventBus()

class WorkflowRequest(BaseModel):
    workflow_id: str
    inputs: dict = {}

@app.post("/api/execute/{workflow_id}")
async def execute_workflow(workflow_id: str, request: WorkflowRequest):
    # In a real system, we would look up the file path from DB
    # For MVP, we assume local file
    file_path = f"workflows/{workflow_id}.bbx"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    try:
        result = await run_file(file_path, event_bus)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
