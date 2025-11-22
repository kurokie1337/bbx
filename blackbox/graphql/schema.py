"""GraphQL schema for BBX."""

from datetime import datetime
from typing import List, Optional

import strawberry


@strawberry.type
class Step:
    id: str
    mcp: str
    method: str
    status: Optional[str] = None
    output: Optional[str] = None


@strawberry.type
class Workflow:
    id: str
    name: str
    version: str
    description: Optional[str] = None
    steps: List[Step]


@strawberry.type
class WorkflowExecution:
    id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    outputs: Optional[str] = None


@strawberry.type
class Query:
    @strawberry.field
    def workflows(self) -> List[Workflow]:
        """Get all workflows."""
        # Implementation
        return []

    @strawberry.field
    def workflow(self, id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        # Implementation
        return None

    @strawberry.field
    def executions(self, workflow_id: Optional[str] = None) -> List[WorkflowExecution]:
        """Get workflow executions."""
        # Implementation
        return []


@strawberry.type
class Mutation:
    @strawberry.mutation
    def execute_workflow(
        self, workflow_id: str, inputs: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute a workflow."""
        # Implementation
        return WorkflowExecution(
            id="exec-1",
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.now(),
        )

    @strawberry.mutation
    def create_workflow(self, workflow_yaml: str) -> Workflow:
        """Create a new workflow."""
        # Implementation
        return Workflow(id="wf-1", name="New", version="6.0", steps=[])


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def workflow_updates(self, workflow_id: str):
        """Subscribe to workflow execution updates."""
        # Implementation
        yield WorkflowExecution(
            id="exec-1",
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.now(),
        )


schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
