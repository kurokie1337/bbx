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
