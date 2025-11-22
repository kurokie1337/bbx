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
BBX Terraform Adapter

Provides Infrastructure as Code automation:
- Terraform init/plan/apply/destroy
- State management
- Variable passing
- Output extraction
- Multi-provider support

Examples:
    # Initialize Terraform
    - id: tf_init
      mcp: bbx.terraform
      method: init
      inputs:
        working_dir: "./terraform"

    # Plan infrastructure
    - id: tf_plan
      mcp: bbx.terraform
      method: plan
      inputs:
        working_dir: "./terraform"
        var_file: "prod.tfvars"

    # Apply changes
    - id: tf_apply
      mcp: bbx.terraform
      method: apply
      inputs:
        working_dir: "./terraform"
        auto_approve: true
"""

import json
from typing import Dict, Any
from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse


class TerraformAdapter(DockerizedAdapter):
    """BBX Adapter for Terraform operations (Dockerized)"""

    def __init__(self):
        super().__init__(
            adapter_name="Terraform",
            docker_image="hashicorp/terraform:latest",
            cli_tool="terraform",
            version_args=["-version"],
            required=True
        )

    def _run_tf_command(self, *args, working_dir=None, timeout=600):
        """Run terraform command with working directory support"""
        # We don't need to manually construct cmd with self.cli_tool here
        # because DockerizedAdapter.run_command handles it (appending args to image)
        
        # working_dir is passed to run_command which sets -w in Docker
        
        return self.run_command(
            *args,
            working_dir=working_dir,
            timeout=timeout,
            output_format="json" if "-json" in args else "text"
        )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Terraform method"""
        self.log_execution(method, inputs)

        handlers = {
            "init": self._init,
            "plan": self._plan,
            "apply": self._apply,
            "destroy": self._destroy,
            "output": self._output,
            "validate": self._validate,
            "fmt": self._fmt,
            "show": self._show,
            "state_list": self._state_list,
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

    # Initialization

    async def _init(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize Terraform working directory

        Inputs:
            working_dir: Path to Terraform configuration
            backend_config: Backend configuration (optional)
            upgrade: Upgrade modules (optional)
        """
        working_dir = inputs.get("working_dir", ".")
        args = ["init"]

        if inputs.get("upgrade"):
            args.append("-upgrade")

        # Backend config
        for key, value in inputs.get("backend_config", {}).items():
            args.extend(["-backend-config", f"{key}={value}"])

        response = self._run_tf_command(*args, working_dir=working_dir)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir},
                status="initialized"
            ).to_dict()

        return response.to_dict()

    # Planning and Apply

    async def _plan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Terraform execution plan

        Inputs:
            working_dir: Path to configuration
            var_file: Variables file (optional)
            variables: Dict of variables (optional)
            out: Save plan to file (optional)
        """
        working_dir = inputs.get("working_dir", ".")
        args = ["plan"]

        # Variable file
        if "var_file" in inputs:
            args.extend(["-var-file", inputs["var_file"]])

        # Individual variables
        for key, value in inputs.get("variables", {}).items():
            args.extend(["-var", f"{key}={value}"])

        # Output file
        if "out" in inputs:
            args.extend(["-out", inputs["out"]])

        # JSON output for parsing
        args.append("-json")

        response = self._run_tf_command(*args, working_dir=working_dir)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir, "plan_output": response.data},
                status="planned"
            ).to_dict()

        return response.to_dict()

    async def _apply(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Terraform changes

        Inputs:
            working_dir: Path to configuration
            auto_approve: Skip confirmation (optional)
            var_file: Variables file (optional)
            variables: Dict of variables (optional)
            plan_file: Use saved plan (optional)
        """
        working_dir = inputs.get("working_dir", ".")
        args = ["apply"]

        # Auto approve
        if inputs.get("auto_approve"):
            args.append("-auto-approve")

        # Plan file
        if "plan_file" in inputs:
            args.append(inputs["plan_file"])
        else:
            # Variable file
            if "var_file" in inputs:
                args.extend(["-var-file", inputs["var_file"]])

            # Individual variables
            for key, value in inputs.get("variables", {}).items():
                args.extend(["-var", f"{key}={value}"])

        response = self._run_tf_command(*args, working_dir=working_dir, timeout=1800)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir},
                status="applied"
            ).to_dict()

        return response.to_dict()

    async def _destroy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Destroy Terraform-managed infrastructure

        Inputs:
            working_dir: Path to configuration
            auto_approve: Skip confirmation (optional)
            var_file: Variables file (optional)
            variables: Dict of variables (optional)
        """
        working_dir = inputs.get("working_dir", ".")
        args = ["destroy"]

        if inputs.get("auto_approve"):
            args.append("-auto-approve")

        if "var_file" in inputs:
            args.extend(["-var-file", inputs["var_file"]])

        for key, value in inputs.get("variables", {}).items():
            args.extend(["-var", f"{key}={value}"])

        response = self._run_tf_command(*args, working_dir=working_dir, timeout=1800)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir},
                status="destroyed"
            ).to_dict()

        return response.to_dict()

    # Outputs and State

    async def _output(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Terraform outputs

        Inputs:
            working_dir: Path to configuration
            name: Specific output name (optional)
            json: Return as JSON (optional)
        """
        working_dir = inputs.get("working_dir", ".")
        args = ["output"]

        if inputs.get("json", True):
            args.append("-json")

        if "name" in inputs:
            args.append(inputs["name"])

        response = self._run_tf_command(*args, working_dir=working_dir)

        if response.success and inputs.get("json", True):
            try:
                data_str = response.data if isinstance(response.data, str) else str(response.data)
                outputs = json.loads(data_str)
                return AdapterResponse.success_response(
                    data={"outputs": outputs, "working_dir": working_dir},
                    status="ok"
                ).to_dict()
            except json.JSONDecodeError:
                return AdapterResponse.error_response(
                    error="Failed to parse outputs"
                ).to_dict()

        if response.success:
            return AdapterResponse.success_response(
                data={"output": response.data, "working_dir": working_dir},
                status="ok"
            ).to_dict()

        return response.to_dict()

    async def _validate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Terraform configuration"""
        working_dir = inputs.get("working_dir", ".")
        response = self._run_tf_command("validate", "-json", working_dir=working_dir)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir},
                status="valid"
            ).to_dict()

        return AdapterResponse.success_response(
            data={"working_dir": working_dir, "errors": response.error},
            status="invalid"
        ).to_dict()

    async def _fmt(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format Terraform files"""
        working_dir = inputs.get("working_dir", ".")
        args = ["fmt"]

        if inputs.get("check"):
            args.append("-check")

        if inputs.get("recursive"):
            args.append("-recursive")

        response = self._run_tf_command(*args, working_dir=working_dir)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir, "formatted_files": response.data},
                status="formatted"
            ).to_dict()

        return response.to_dict()

    async def _show(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show Terraform state or plan"""
        working_dir = inputs.get("working_dir", ".")
        args = ["show", "-json"]

        if "plan_file" in inputs:
            args.append(inputs["plan_file"])

        response = self._run_tf_command(*args, working_dir=working_dir)

        if response.success:
            return AdapterResponse.success_response(
                data={"working_dir": working_dir, "state": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    async def _state_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List resources in Terraform state"""
        working_dir = inputs.get("working_dir", ".")
        response = self._run_tf_command("state", "list", working_dir=working_dir)

        if response.success:
            data_str = response.data if isinstance(response.data, str) else str(response.data)
            resources = data_str.split('\n') if data_str else []
            return AdapterResponse.success_response(
                data={
                    "resources": resources,
                    "count": len(resources),
                    "working_dir": working_dir
                },
                status="ok"
            ).to_dict()

        return response.to_dict()
