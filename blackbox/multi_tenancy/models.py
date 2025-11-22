"""Multi-tenancy models."""

from typing import Optional

from pydantic import BaseModel


class Tenant(BaseModel):
    id: str
    name: str
    namespace: str
    quota_cpu: Optional[int] = None
    quota_memory: Optional[int] = None
    quota_workflows: Optional[int] = None


class TenantContext:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    def get_namespace(self):
        return f"bbx-tenant-{self.tenant_id}"
