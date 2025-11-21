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
BBX GCP Adapter

Provides complete Google Cloud Platform automation:
- Compute Engine (VMs)
- Cloud Storage (buckets)
- Cloud Functions
- Cloud Run
- GKE (Kubernetes Engine)
- Cloud SQL

Examples:
    # Create VM instance
    - id: create_vm
      mcp: bbx.gcp
      method: compute_create
      inputs:
        name: "my-vm"
        machine_type: "e2-micro"
        zone: "us-central1-a"

    # Deploy Cloud Function
    - id: deploy_function
      mcp: bbx.gcp
      method: function_deploy
      inputs:
        name: "my-function"
        runtime: "python311"
        entry_point: "main"
        source: "./function"
"""

import json
from typing import Dict, Any
from blackbox.core.base_adapter import CLIAdapter, AdapterResponse


class GCPAdapter(CLIAdapter):
    """BBX Adapter for Google Cloud Platform using gcloud CLI"""

    def __init__(self):
        super().__init__(
            adapter_name="GCP",
            cli_tool="gcloud",
            version_args=["version"],
            required=True
        )

    def _run_gcloud(self, *args, **kwargs):
        """Run gcloud command with format=json"""
        # Override to add --format=json instead of --output json
        return super().run_command(*args, **kwargs)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute GCP method"""
        self.log_execution(method, inputs)

        handlers = {
            # Compute Engine
            "compute_create": self._compute_create,
            "compute_delete": self._compute_delete,
            "compute_list": self._compute_list,
            # Cloud Storage
            "storage_create_bucket": self._storage_create_bucket,
            "storage_upload": self._storage_upload,
            "storage_list": self._storage_list,
            # Cloud Functions
            "function_deploy": self._function_deploy,
            "function_call": self._function_call,
            # Cloud Run
            "run_deploy": self._run_deploy,
            # GKE
            "gke_create": self._gke_create,
            "gke_get_credentials": self._gke_get_credentials,
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

    # Override run_command to use --format=json instead of --output json
    def run_command(self, *args, **kwargs):
        """Execute gcloud command with --format=json"""
        # Remove --output json if present and add --format=json
        args_list = list(args)

        # Don't add --output json for gcloud
        if kwargs.get('output_format') == 'json':
            if '--format=json' not in args_list:
                args_list.append('--format=json')
            kwargs.pop('output_format', None)

        # Call parent without output_format
        return super().run_command(*args_list, **kwargs)

    # Compute Engine Operations

    async def _compute_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create Compute Engine instance"""
        name = inputs.get("name")
        if not name:
            return AdapterResponse.error_response(error="name is required").to_dict()

        zone = inputs.get("zone", "us-central1-a")
        machine_type = inputs.get("machine_type", "e2-micro")

        args = [
            "compute", "instances", "create", name,
            "--zone", zone,
            "--machine-type", machine_type,
            "--format=json"
        ]

        if "image_family" in inputs:
            args.extend(["--image-family", inputs["image_family"]])
        if "image_project" in inputs:
            args.extend(["--image-project", inputs["image_project"]])

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"instance": name, "zone": zone},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _compute_delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Compute Engine instance"""
        name = inputs.get("name")
        if not name:
            return AdapterResponse.error_response(error="name is required").to_dict()

        zone = inputs.get("zone", "us-central1-a")

        args = [
            "compute", "instances", "delete", name,
            "--zone", zone,
            "--quiet",
            "--format=json"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"instance": name},
                status="deleted"
            ).to_dict()

        return response.to_dict()

    async def _compute_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List Compute Engine instances"""
        args = ["compute", "instances", "list", "--format=json"]

        if "zone" in inputs:
            args.extend(["--zones", inputs["zone"]])

        response = self.run_command(*args, output_format="json")

        if response.success:
            instances = response.data if isinstance(response.data, list) else []
            return AdapterResponse.success_response(
                data={"instances": instances, "count": len(instances)},
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Cloud Storage Operations

    async def _storage_create_bucket(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create Cloud Storage bucket"""
        bucket_name = inputs.get("bucket_name")
        if not bucket_name:
            return AdapterResponse.error_response(error="bucket_name is required").to_dict()

        location = inputs.get("location", "US")

        args = [
            "storage", "buckets", "create", f"gs://{bucket_name}",
            "--location", location,
            "--format=json"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"bucket": bucket_name, "location": location},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _storage_upload(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Upload file to Cloud Storage"""
        bucket = inputs.get("bucket")
        source = inputs.get("source")

        if not all([bucket, source]):
            return AdapterResponse.error_response(
                error="bucket and source are required"
            ).to_dict()

        destination = inputs.get("destination", "")

        args = [
            "storage", "cp", source, f"gs://{bucket}/{destination}",
            "--format=json"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"bucket": bucket, "destination": destination},
                status="uploaded"
            ).to_dict()

        return response.to_dict()

    async def _storage_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List Cloud Storage objects"""
        bucket = inputs.get("bucket")
        if not bucket:
            return AdapterResponse.error_response(error="bucket is required").to_dict()

        prefix = inputs.get("prefix", "")

        args = ["storage", "ls", f"gs://{bucket}/{prefix}", "--format=json"]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Cloud Functions Operations

    async def _function_deploy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy Cloud Function"""
        name = inputs.get("name")
        if not name:
            return AdapterResponse.error_response(error="name is required").to_dict()

        runtime = inputs.get("runtime", "python311")
        entry_point = inputs.get("entry_point", "main")
        source = inputs.get("source", ".")

        args = [
            "functions", "deploy", name,
            "--runtime", runtime,
            "--entry-point", entry_point,
            "--source", source,
            "--trigger-http",
            "--allow-unauthenticated",
            "--format=json"
        ]

        if "region" in inputs:
            args.extend(["--region", inputs["region"]])

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"function": name},
                status="deployed"
            ).to_dict()

        return response.to_dict()

    async def _function_call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call Cloud Function"""
        name = inputs.get("name")
        if not name:
            return AdapterResponse.error_response(error="name is required").to_dict()

        data = inputs.get("data", {})

        args = ["functions", "call", name, "--format=json"]

        if data:
            args.extend(["--data", json.dumps(data)])

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="called"
            ).to_dict()

        return response.to_dict()

    # Cloud Run Operations

    async def _run_deploy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Cloud Run"""
        service = inputs.get("service")
        image = inputs.get("image")

        if not all([service, image]):
            return AdapterResponse.error_response(
                error="service and image are required"
            ).to_dict()

        region = inputs.get("region", "us-central1")

        args = [
            "run", "deploy", service,
            "--image", image,
            "--region", region,
            "--allow-unauthenticated",
            "--format=json"
        ]

        if "port" in inputs:
            args.extend(["--port", str(inputs["port"])])

        response = self.run_command(*args, timeout=600, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"service": service, "region": region},
                status="deployed"
            ).to_dict()

        return response.to_dict()

    # GKE Operations

    async def _gke_create(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create GKE cluster"""
        cluster_name = inputs.get("cluster_name")
        if not cluster_name:
            return AdapterResponse.error_response(error="cluster_name is required").to_dict()

        zone = inputs.get("zone", "us-central1-a")
        num_nodes = inputs.get("num_nodes", 3)

        args = [
            "container", "clusters", "create", cluster_name,
            "--zone", zone,
            "--num-nodes", str(num_nodes),
            "--format=json"
        ]

        if "machine_type" in inputs:
            args.extend(["--machine-type", inputs["machine_type"]])

        response = self.run_command(*args, timeout=900, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data={"cluster": cluster_name, "zone": zone},
                status="created"
            ).to_dict()

        return response.to_dict()

    async def _gke_get_credentials(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get GKE cluster credentials"""
        cluster_name = inputs.get("cluster_name")
        if not cluster_name:
            return AdapterResponse.error_response(error="cluster_name is required").to_dict()

        zone = inputs.get("zone", "us-central1-a")

        args = [
            "container", "clusters", "get-credentials", cluster_name,
            "--zone", zone,
            "--format=json"
        ]

        response = self.run_command(*args, output_format="json")

        if response.success:
            return AdapterResponse.success_response(
                data=response.data,
                status="ok"
            ).to_dict()

        return response.to_dict()
