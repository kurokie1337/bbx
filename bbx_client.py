import logging
import httpx
from typing import Optional, Dict, Any, Union
from uuid import UUID

from config import settings
from bbx_models import (
    AuthLogin, 
    AuthResponse, 
    WorkflowCreate, 
    WorkflowResponse, 
    WorkflowListResponse,
    WorkflowUpdate,
    ExecutionCreate,
    ExecutionResponse
)

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

class BlackboxClient:
    def __init__(self, base_url: str = settings.BLACKBOX_API_URL, timeout: int = settings.BLACKBOX_API_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.token: Optional[str] = None
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def authenticate(self, username: str = settings.BLACKBOX_USERNAME, password: str = settings.BLACKBOX_PASSWORD) -> AuthResponse:
        """Authenticate with BLACKBOX API and store the token."""
        logger.info(f"Authenticating as {username}...")
        payload = AuthLogin(username=username, password=password)
        
        try:
            response = self.client.post("/api/auth/login", json=payload.model_dump())
            response.raise_for_status()
            
            auth_data = AuthResponse(**response.json())
            self.token = auth_data.access_token
            self._update_headers()
            
            logger.info("Authentication successful")
            return auth_data
        except httpx.HTTPStatusError as e:
            logger.error(f"Authentication failed: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise

    def _update_headers(self):
        """Update client headers with the current token."""
        if self.token:
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})

    def _ensure_auth(self):
        """Ensure the client is authenticated."""
        if not self.token:
            self.authenticate()

    def create_workflow(self, workflow: WorkflowCreate) -> WorkflowResponse:
        """Create a new workflow."""
        self._ensure_auth()
        logger.info(f"Creating workflow: {workflow.name}")
        
        response = self.client.post("/api/workflows", json=workflow.model_dump())
        response.raise_for_status()
        
        # The API returns { "status": "success", "data": { ... } }
        data = response.json().get("data")
        return WorkflowResponse(**data)

    def get_workflow(self, workflow_id: Union[str, UUID]) -> WorkflowResponse:
        """Get a workflow by ID."""
        self._ensure_auth()
        logger.info(f"Getting workflow: {workflow_id}")
        
        response = self.client.get(f"/api/workflows/{workflow_id}")
        response.raise_for_status()
        
        data = response.json().get("data")
        return WorkflowResponse(**data)

    def list_workflows(self, skip: int = 0, limit: int = 100) -> WorkflowListResponse:
        """List workflows with pagination."""
        self._ensure_auth()
        logger.info(f"Listing workflows (skip={skip}, limit={limit})")
        
        response = self.client.get("/api/workflows", params={"skip": skip, "limit": limit})
        response.raise_for_status()
        
        # Assuming the API returns a list or a paginated object. 
        # Adjusting based on standard BLACKBOX response format if needed.
        # If API returns list directly:
        items_data = response.json()
        if isinstance(items_data, dict) and "items" in items_data:
             return WorkflowListResponse(**items_data)
        elif isinstance(items_data, list):
             # Mocking total count if API just returns a list
             items = [WorkflowResponse(**item) for item in items_data]
             return WorkflowListResponse(items=items, total=len(items), page=1, size=limit)
        
        # Fallback
        return WorkflowListResponse(items=[], total=0, page=1, size=limit)

    def update_workflow(self, workflow_id: Union[str, UUID], update_data: WorkflowUpdate) -> WorkflowResponse:
        """Update an existing workflow."""
        self._ensure_auth()
        logger.info(f"Updating workflow: {workflow_id}")
        
        response = self.client.put(
            f"/api/workflows/{workflow_id}", 
            json=update_data.model_dump(exclude_unset=True)
        )
        response.raise_for_status()
        
        data = response.json().get("data")
        return WorkflowResponse(**data)

    def delete_workflow(self, workflow_id: Union[str, UUID]) -> bool:
        """Delete a workflow."""
        self._ensure_auth()
        logger.info(f"Deleting workflow: {workflow_id}")
        
        response = self.client.delete(f"/api/workflows/{workflow_id}")
        response.raise_for_status()
        return True

    def execute_workflow(self, workflow_id: Union[str, UUID], trigger_data: Dict[str, Any] = {}) -> ExecutionResponse:
        """Execute a workflow."""
        self._ensure_auth()
        logger.info(f"Executing workflow: {workflow_id}")
        
        payload = ExecutionCreate(workflow_id=workflow_id, trigger_data=trigger_data)
        response = self.client.post("/api/executions", json=payload.model_dump())
        response.raise_for_status()
        
        data = response.json().get("data")
        return ExecutionResponse(**data)

    def get_execution_status(self, execution_id: str) -> ExecutionResponse:
        """Get execution status."""
        self._ensure_auth()
        
        response = self.client.get(f"/api/executions/{execution_id}")
        response.raise_for_status()
        
        data = response.json().get("data")
        return ExecutionResponse(**data)

    def health_check(self) -> bool:
        """Check if API is reachable."""
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
