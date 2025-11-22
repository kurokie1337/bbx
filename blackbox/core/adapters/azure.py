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
BBX Azure Adapter

Provides complete Microsoft Azure automation:
- Virtual Machines
- Storage Accounts
- Azure Functions
- AKS (Azure Kubernetes Service)
- Azure SQL
- Container Instances

Examples:
    # Create VM
    - id: create_vm
      mcp: bbx.azure
      method: vm_create
      inputs:
        name: "my-vm"
        resource_group: "my-rg"
        image: "UbuntuLTS"

    # Deploy Function App
    - id: deploy_function
      mcp: bbx.azure
      method: function_deploy
      inputs:
        name: "my-function-app"
        resource_group: "my-rg"
        runtime: "python"
"""

import os
from typing import Dict, Any
from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse


class AzureAdapter(DockerizedAdapter):
    """BBX Adapter for Microsoft Azure using az CLI (Dockerized)"""

    def __init__(self):
        super().__init__(
            adapter_name="Azure",
            docker_image="mcr.microsoft.com/azure-cli",
            cli_tool="az",
            version_args=["version"],
            required=True
        )

    def run_command(self, *args, **kwargs):
        """Override run_command to inject Azure credentials"""
        env = kwargs.get("env", {}) or {}
        
        # Pass Azure credentials from host environment
        azure_vars = [
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "AZURE_TENANT_ID",
            "AZURE_SUBSCRIPTION_ID"
        ]
        
        for var in azure_vars:
            if os.environ.get(var):
                env[var] = os.environ[var]
                
        kwargs["env"] = env
        return super().run_command(*args, **kwargs)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Azure method"""
        self.log_execution(method, inputs)

        handlers = {
            # VM operations
            "vm_create": self._vm_create,
            "vm_delete": self._vm_delete,
            "vm_list": self._vm_list,
            # Storage
            "storage_create_account": self._storage_create_account,
            "storage_upload_blob": self._storage_upload_blob,
            # Functions
            "function_create_app": self._function_create_app,
            "function_deploy": self._function_deploy,
            # AKS
            "aks_create": self._aks_create,
            "aks_get_credentials": self._aks_get_credentials,
            # Resource Groups
            "group_create": self._group_create,
        }

        handler = handlers.get(method)
        if not handler:
            return AdapterResponse.error_response(
                error=f"Unknown method: {method}"
            ).to_dict()

        try:
            result = await handler(inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            return AdapterResponse.error_response(error=str(e)).to_dict()

    # VM Operations

    async def _vm_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create Azure VM"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        image = inputs.get("image", "UbuntuLTS")
        size = inputs.get("size", "Standard_B1s")

        args = [
            "vm", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--image", image,
            "--size", size,
            "--generate-ssh-keys"
        ]

        if "admin_username" in inputs:
            args.extend(["--admin-username", inputs["admin_username"]])

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"vm": name, "resource_group": resource_group},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _vm_delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Azure VM"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        args = [
            "vm", "delete",
            "--name", name,
            "--resource-group", resource_group,
            "--yes"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"vm": name},
                status="deleted"
            ).to_dict()

        return response.to_dict()

    async def _vm_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List Azure VMs"""
        args = ["vm", "list"]

        if "resource_group" in inputs:
            args.extend(["--resource-group", inputs["resource_group"]])

        response = self.run_command(*args, output_format="json")

        if response.success:
            vms = response.data if isinstance(response.data, list) else []
            return AdapterResponse.success_response(
                data={"vms": vms, "count": len(vms)},
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Storage Operations

    async def _storage_create_account(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create Storage Account"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        location = inputs.get("location", "eastus")

        args = [
            "storage", "account", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--location", location,
            "--sku", "Standard_LRS"
        ]

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"account": name, "location": location},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _storage_upload_blob(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Upload blob to storage"""
        account_name = inputs.get("account_name")
        container = inputs.get("container")
        file_path = inputs.get("file")

        if not all([account_name, container, file_path]):
            return AdapterResponse.error_response(
                error="account_name, container, and file are required"
            ).to_dict()

        from pathlib import Path
        blob_name = inputs.get("blob_name", Path(file_path).name)

        args = [
            "storage", "blob", "upload",
            "--account-name", account_name,
            "--container-name", container,
            "--name", blob_name,
            "--file", file_path
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"blob": blob_name, "container": container},
                status="uploaded"
            ).to_dict()

        return response.to_dict()

    # Function App Operations

    async def _function_create_app(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create Function App"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")
        storage_account = inputs.get("storage_account")

        if not all([name, resource_group, storage_account]):
            return AdapterResponse.error_response(
                error="name, resource_group, and storage_account are required"
            ).to_dict()

        runtime = inputs.get("runtime", "python")
        location = inputs.get("location", "eastus")

        args = [
            "functionapp", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--storage-account", storage_account,
            "--runtime", runtime,
            "--consumption-plan-location", location
        ]

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"function_app": name, "runtime": runtime},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _function_deploy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Function App"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        source = inputs.get("source", ".")

        # Note: This typically uses 'func azure functionapp publish'
        # but we'll use az for consistency
        args = [
            "functionapp", "deployment", "source", "config-zip",
            "--name", name,
            "--resource-group", resource_group,
            "--src", source
        ]

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"function_app": name},
                status="deployed"
            ).to_dict()

        return response.to_dict()

    # AKS Operations

    async def _aks_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create AKS cluster"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        node_count = inputs.get("node_count", 3)

        args = [
            "aks", "create",
            "--name", name,
            "--resource-group", resource_group,
            "--node-count", str(node_count),
            "--generate-ssh-keys"
        ]

        if "vm_size" in inputs:
            args.extend(["--node-vm-size", inputs["vm_size"]])

        response = self.run_command(*args, timeout=1200, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"cluster": name, "nodes": node_count},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _aks_get_credentials(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get AKS credentials"""
        name = inputs.get("name")
        resource_group = inputs.get("resource_group")

        if not all([name, resource_group]):
            return AdapterResponse.error_response(
                error="name and resource_group are required"
            ).to_dict()

        args = [
            "aks", "get-credentials",
            "--name", name,
            "--resource-group", resource_group,
            "--overwrite-existing"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Resource Group Operations

    async def _group_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource group"""
        name = inputs.get("name")

        if not name:
            return AdapterResponse.error_response(
                error="name is required"
            ).to_dict()

        location = inputs.get("location", "eastus")

        args = [
            "group", "create",
            "--name", name,
            "--location", location
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"resource_group": name, "location": location},
                status="created"
            ).to_dict()

        return response.to_dict()
