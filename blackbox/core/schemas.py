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
BBX Pydantic Schemas for Input Validation

Provides type-safe validation for all adapter inputs using Pydantic models.
"""

from typing import Dict, Any, List, Optional, Union, Annotated
from pydantic import BaseModel, Field, validator
try:
    from pydantic import conint
except ImportError:
    # Pydantic v2 doesn't have conint, use Annotated instead
    conint = None  # type: ignore


# ============================================================================
# AWS Adapter Schemas
# ============================================================================

class EC2LaunchInput(BaseModel):
    """Input schema for EC2 instance launch"""
    image_id: str = Field(..., description="AMI image ID")
    instance_type: str = Field(default="t2.micro", description="Instance type")
    key_name: Optional[str] = Field(None, description="SSH key pair name")
    security_groups: Optional[List[str]] = Field(None, description="Security group IDs")
    subnet_id: Optional[str] = Field(None, description="Subnet ID")
    user_data: Optional[str] = Field(None, description="User data script")
    tags: Optional[Dict[str, str]] = Field(None, description="Instance tags")

    @validator('instance_type')
    def validate_instance_type(cls, v):
        """Validate instance type format"""
        if not v.startswith(('t2.', 't3.', 'm5.', 'c5.', 'r5.')):
            raise ValueError(f"Invalid instance type: {v}")
        return v


class S3UploadInput(BaseModel):
    """Input schema for S3 file upload"""
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="Object key")
    file_path: str = Field(..., description="Local file path")
    acl: Optional[str] = Field("private", description="ACL setting")
    metadata: Optional[Dict[str, str]] = Field(None, description="Object metadata")

    @validator('bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if not v or len(v) < 3 or len(v) > 63:
            raise ValueError("Bucket name must be between 3 and 63 characters")
        return v.lower()


# ============================================================================
# GCP Adapter Schemas
# ============================================================================

class GCPComputeCreateInput(BaseModel):
    """Input schema for GCP Compute Engine instance"""
    name: str = Field(..., description="Instance name")
    zone: str = Field(default="us-central1-a", description="GCP zone")
    machine_type: str = Field(default="e2-micro", description="Machine type")
    image_family: Optional[str] = Field(None, description="Image family")
    image_project: Optional[str] = Field(None, description="Image project")
    tags: Optional[List[str]] = Field(None, description="Network tags")

    @validator('name')
    def validate_name(cls, v):
        """Validate GCP resource name"""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Name must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class GCPStorageUploadInput(BaseModel):
    """Input schema for GCP Storage upload"""
    bucket: str = Field(..., description="GCS bucket name")
    source: str = Field(..., description="Source file path")
    destination: str = Field(default="", description="Destination path in bucket")


# ============================================================================
# Azure Adapter Schemas
# ============================================================================

class AzureVMCreateInput(BaseModel):
    """Input schema for Azure VM creation"""
    name: str = Field(..., description="VM name")
    resource_group: str = Field(..., description="Resource group name")
    image: str = Field(default="UbuntuLTS", description="VM image")
    size: str = Field(default="Standard_B1s", description="VM size")
    admin_username: Optional[str] = Field(None, description="Admin username")
    location: str = Field(default="eastus", description="Azure region")


class AzureStorageAccountInput(BaseModel):
    """Input schema for Azure Storage Account"""
    name: str = Field(..., description="Storage account name")
    resource_group: str = Field(..., description="Resource group name")
    location: str = Field(default="eastus", description="Azure region")
    sku: str = Field(default="Standard_LRS", description="Storage SKU")

    @validator('name')
    def validate_storage_name(cls, v):
        """Validate storage account name"""
        if not v.isalnum() or len(v) < 3 or len(v) > 24:
            raise ValueError("Storage account name must be 3-24 alphanumeric characters")
        return v.lower()


# ============================================================================
# Docker Adapter Schemas
# ============================================================================

class DockerRunInput(BaseModel):
    """Input schema for Docker container run"""
    image: str = Field(..., description="Docker image name")
    name: Optional[str] = Field(None, description="Container name")
    ports: Optional[List[str]] = Field(None, description="Port mappings (e.g., '8080:80')")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    volumes: Optional[List[str]] = Field(None, description="Volume mounts")
    command: Optional[str] = Field(None, description="Command to run")
    detach: bool = Field(True, description="Run in background")


class DockerBuildInput(BaseModel):
    """Input schema for Docker image build"""
    path: str = Field(..., description="Build context path")
    tag: str = Field(..., description="Image tag")
    dockerfile: Optional[str] = Field(None, description="Dockerfile path")
    build_args: Optional[Dict[str, str]] = Field(None, description="Build arguments")


# ============================================================================
# Kubernetes Adapter Schemas
# ============================================================================

class K8sApplyInput(BaseModel):
    """Input schema for Kubernetes apply"""
    file: str = Field(..., description="Manifest file path")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace")
    dry_run: bool = Field(False, description="Dry run mode")


class K8sScaleInput(BaseModel):
    """Input schema for Kubernetes scale"""
    deployment: str = Field(..., description="Deployment name")
    replicas: Annotated[int, Field(ge=0)] = Field(..., description="Number of replicas")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace")


class K8sGetInput(BaseModel):
    """Input schema for Kubernetes get"""
    resource: str = Field(..., description="Resource type (e.g., 'pods', 'services')")
    name: Optional[str] = Field(None, description="Resource name")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace")
    all_namespaces: bool = Field(False, description="Search all namespaces")
    output: str = Field("json", description="Output format")


# ============================================================================
# Terraform Adapter Schemas
# ============================================================================

class TerraformInitInput(BaseModel):
    """Input schema for Terraform init"""
    working_dir: str = Field(default=".", description="Working directory")
    upgrade: bool = Field(False, description="Upgrade modules")
    backend_config: Optional[Dict[str, str]] = Field(None, description="Backend configuration")


class TerraformApplyInput(BaseModel):
    """Input schema for Terraform apply"""
    working_dir: str = Field(default=".", description="Working directory")
    auto_approve: bool = Field(False, description="Skip confirmation")
    var_file: Optional[str] = Field(None, description="Variables file")
    variables: Optional[Dict[str, Any]] = Field(None, description="Inline variables")
    plan_file: Optional[str] = Field(None, description="Plan file to apply")


# ============================================================================
# Ansible Adapter Schemas
# ============================================================================

class AnsiblePlaybookInput(BaseModel):
    """Input schema for Ansible playbook"""
    playbook: str = Field(..., description="Playbook file path")
    inventory: Optional[str] = Field(None, description="Inventory file")
    limit: Optional[str] = Field(None, description="Limit to hosts")
    tags: Optional[Union[str, List[str]]] = Field(None, description="Tags to run")
    skip_tags: Optional[Union[str, List[str]]] = Field(None, description="Tags to skip")
    extra_vars: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Extra variables")
    check: bool = Field(False, description="Dry run mode")
    diff: bool = Field(False, description="Show diffs")
    verbose: Annotated[int, Field(ge=0, le=4)] = Field(0, description="Verbosity level (0-4)")


class AnsibleAdhocInput(BaseModel):
    """Input schema for Ansible ad-hoc command"""
    pattern: str = Field(..., description="Host pattern")
    module: str = Field(..., description="Module name")
    args: Optional[str] = Field(None, description="Module arguments")
    inventory: Optional[str] = Field(None, description="Inventory file")
    become: bool = Field(False, description="Use privilege escalation")


# ============================================================================
# Workflow Schemas
# ============================================================================

class WorkflowStepInput(BaseModel):
    """Input schema for workflow step"""
    id: str = Field(..., description="Step ID")
    mcp: str = Field(..., description="Adapter name")
    method: str = Field(..., description="Method to execute")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Method inputs")
    when: Optional[str] = Field(None, description="Conditional expression")
    retry: Annotated[int, Field(ge=0)] = Field(0, description="Retry count")
    retry_delay: Annotated[int, Field(ge=0)] = Field(1000, description="Retry delay (ms)")
    timeout: Annotated[int, Field(ge=0)] = Field(30000, description="Timeout (ms)")
    outputs: Optional[str] = Field(None, description="Output variable name")


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    id: str = Field(..., description="Workflow ID")
    name: Optional[str] = Field(None, description="Workflow name")
    version: str = Field(default="6.0", description="BBX version")
    description: Optional[str] = Field(None, description="Workflow description")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Workflow inputs")
    steps: List[WorkflowStepInput] = Field(..., description="Workflow steps")
    outputs: Optional[Dict[str, Any]] = Field(None, description="Workflow outputs")


# ============================================================================
# Validation Helper Functions
# ============================================================================

def validate_input(schema_class: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """
    Validate input data against a Pydantic schema

    Args:
        schema_class: Pydantic model class
        data: Input data to validate

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails
    """
    return schema_class(**data)


def validate_and_dict(schema_class: type[BaseModel], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input data and return as dictionary

    Args:
        schema_class: Pydantic model class
        data: Input data to validate

    Returns:
        Validated data as dictionary

    Raises:
        ValidationError: If validation fails
    """
    model = schema_class(**data)
    return model.dict(exclude_none=True)


# ============================================================================
# Schema Registry
# ============================================================================

ADAPTER_SCHEMAS: Dict[str, type[BaseModel]] = {
    # AWS
    "aws.ec2_launch": EC2LaunchInput,
    "aws.s3_upload": S3UploadInput,

    # GCP
    "gcp.compute_create": GCPComputeCreateInput,
    "gcp.storage_upload": GCPStorageUploadInput,

    # Azure
    "azure.vm_create": AzureVMCreateInput,
    "azure.storage_create_account": AzureStorageAccountInput,

    # Docker
    "docker.run": DockerRunInput,
    "docker.build": DockerBuildInput,

    # Kubernetes
    "k8s.apply": K8sApplyInput,
    "k8s.scale": K8sScaleInput,
    "k8s.get": K8sGetInput,

    # Terraform
    "terraform.init": TerraformInitInput,
    "terraform.apply": TerraformApplyInput,

    # Ansible
    "ansible.playbook": AnsiblePlaybookInput,
    "ansible.adhoc": AnsibleAdhocInput,
}


def get_schema(adapter: str, method: str) -> Optional[type[BaseModel]]:
    """
    Get Pydantic schema for adapter method

    Args:
        adapter: Adapter name (e.g., "aws", "gcp")
        method: Method name (e.g., "ec2_launch")

    Returns:
        Pydantic model class or None if not found
    """
    key = f"{adapter}.{method}"
    return ADAPTER_SCHEMAS.get(key)
