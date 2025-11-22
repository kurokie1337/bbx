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
BBX Ansible Adapter

Provides configuration management automation:
- Playbook execution
- Inventory management
- Ad-hoc commands
- Galaxy role installation
- Vault operations

Examples:
    # Run playbook
    - id: deploy
      mcp: bbx.ansible
      method: playbook
      inputs:
        playbook: "deploy.yml"
        inventory: "hosts.ini"
        extra_vars:
          app_version: "1.2.3"

    # Ad-hoc command
    - id: ping_all
      mcp: bbx.ansible
      method: adhoc
      inputs:
        pattern: "all"
        module: "ping"
        inventory: "hosts.ini"
"""

import json
from pathlib import Path
from typing import Dict, Any
from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse


class AnsibleAdapter(DockerizedAdapter):
    """BBX Adapter for Ansible operations (Dockerized)"""

    def __init__(self):
        super().__init__(
            adapter_name="Ansible",
            docker_image="willhallonline/ansible:latest",
            cli_tool="ansible",
            version_args=["--version"],
            required=True
        )

    def run_command(self, *args, **kwargs):
        """Override run_command to mount SSH keys"""
        volumes = kwargs.get("volumes", {}) or {}
        
        # Mount ~/.ssh if it exists
        home = Path.home()
        ssh_dir = home / ".ssh"
        if ssh_dir.exists():
            # Mount to /root/.ssh (container runs as root usually)
            volumes[str(ssh_dir)] = "/root/.ssh"
            
        kwargs["volumes"] = volumes
        return super().run_command(*args, **kwargs)

    def _run_ansible_cmd(self, *args, timeout=600):
        """Run generic ansible command using Dockerized execution"""
        # args is the full command list e.g. ["ansible-playbook", ...]
        # DockerizedAdapter appends args to image
        return self.run_command(*args, timeout=timeout)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Ansible method"""
        self.log_execution(method, inputs)

        handlers = {
            "playbook": self._playbook,
            "adhoc": self._adhoc,
            "galaxy_install": self._galaxy_install,
            "inventory_list": self._inventory_list,
            "vault_encrypt": self._vault_encrypt,
            "vault_decrypt": self._vault_decrypt,
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

    # Playbook Execution

    async def _playbook(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Ansible playbook

        Inputs:
            playbook: Path to playbook file
            inventory: Inventory file/string
            limit: Limit to specific hosts (optional)
            tags: Tags to run (optional)
            skip_tags: Tags to skip (optional)
            extra_vars: Extra variables (optional)
            check: Dry run mode (optional)
            diff: Show diffs (optional)
            verbose: Verbosity level 0-4 (optional)
        """
        playbook = inputs.get("playbook")
        if not playbook:
            return AdapterResponse.error_response(
                error="playbook is required"
            ).to_dict()

        cmd = ["ansible-playbook"]

        # Inventory
        if "inventory" in inputs:
            cmd.extend(["-i", inputs["inventory"]])

        # Limit hosts
        if "limit" in inputs:
            cmd.extend(["--limit", inputs["limit"]])

        # Tags
        if "tags" in inputs:
            tags = inputs["tags"] if isinstance(inputs["tags"], str) else ",".join(inputs["tags"])
            cmd.extend(["--tags", tags])

        if "skip_tags" in inputs:
            skip = inputs["skip_tags"] if isinstance(inputs["skip_tags"], str) else ",".join(inputs["skip_tags"])
            cmd.extend(["--skip-tags", skip])

        # Extra vars
        if "extra_vars" in inputs:
            extra_vars = inputs["extra_vars"]
            if isinstance(extra_vars, dict):
                cmd.extend(["-e", json.dumps(extra_vars)])
            else:
                cmd.extend(["-e", extra_vars])

        # Check mode
        if inputs.get("check"):
            cmd.append("--check")

        # Diff
        if inputs.get("diff"):
            cmd.append("--diff")

        # Verbosity
        verbose = inputs.get("verbose", 0)
        if verbose > 0:
            cmd.append("-" + "v" * min(verbose, 4))

        # Playbook file
        cmd.append(playbook)

        response = self._run_ansible_cmd(*cmd, timeout=1800)

        if response.success:
            return AdapterResponse.success_response(
                data={"playbook": playbook, "output": response.data},
                status="completed"
            ).to_dict()

        return response.to_dict()

    # Ad-hoc Commands

    async def _adhoc(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ad-hoc Ansible command

        Inputs:
            pattern: Host pattern (e.g., "all", "webservers")
            module: Module to run
            args: Module arguments (optional)
            inventory: Inventory file (optional)
            become: Use sudo (optional)
        """
        pattern = inputs.get("pattern")
        module = inputs.get("module")

        if not all([pattern, module]):
            return AdapterResponse.error_response(
                error="pattern and module are required"
            ).to_dict()

        cmd = ["ansible", pattern]

        # Inventory
        if "inventory" in inputs:
            cmd.extend(["-i", inputs["inventory"]])

        # Module
        cmd.extend(["-m", module])

        # Module args
        if "args" in inputs:
            cmd.extend(["-a", inputs["args"]])

        # Become
        if inputs.get("become"):
            cmd.append("--become")

        response = self._run_ansible_cmd(*cmd)

        if response.success:
            return AdapterResponse.success_response(
                data={
                    "pattern": pattern,
                    "module": module,
                    "output": response.data
                },
                status="completed"
            ).to_dict()

        return response.to_dict()

    # Galaxy Operations

    async def _galaxy_install(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install Ansible Galaxy roles/collections

        Inputs:
            name: Role/collection name or requirements file
            type: "role" or "collection" (default: role)
            force: Force reinstall (optional)
        """
        name = inputs.get("name")
        if not name:
            return AdapterResponse.error_response(
                error="name is required"
            ).to_dict()

        install_type = inputs.get("type", "role")

        cmd = ["ansible-galaxy", install_type, "install"]

        if inputs.get("force"):
            cmd.append("--force")

        # Requirements file or direct name
        if name.endswith(".yml") or name.endswith(".yaml"):
            cmd.extend(["-r", name])
        else:
            cmd.append(name)

        response = self._run_ansible_cmd(*cmd)

        if response.success:
            return AdapterResponse.success_response(
                data={"name": name, "type": install_type},
                status="installed"
            ).to_dict()

        return response.to_dict()

    # Inventory Operations

    async def _inventory_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        List inventory hosts

        Inputs:
            inventory: Inventory file
            format: Output format (json/yaml/toml)
        """
        inventory = inputs.get("inventory")
        if not inventory:
            return AdapterResponse.error_response(
                error="inventory is required"
            ).to_dict()

        cmd = ["ansible-inventory", "-i", inventory, "--list"]

        response = self._run_ansible_cmd(*cmd)

        if response.success:
            try:
                data_str = response.data if isinstance(response.data, str) else str(response.data)
                inventory_data = json.loads(data_str)
                return AdapterResponse.success_response(
                    data={
                        "inventory": inventory_data,
                        "hosts": list(inventory_data.get("_meta", {}).get("hostvars", {}).keys())
                    },
                    status="ok"
                ).to_dict()
            except json.JSONDecodeError:
                pass

        if response.success:
            return AdapterResponse.success_response(
                data={"output": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Vault Operations

    async def _vault_encrypt(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt file with Ansible Vault

        Inputs:
            file: File to encrypt
            vault_password_file: Password file (optional)
        """
        file_path = inputs.get("file")
        if not file_path:
            return AdapterResponse.error_response(
                error="file is required"
            ).to_dict()

        cmd = ["ansible-vault", "encrypt", file_path]

        if "vault_password_file" in inputs:
            cmd.extend(["--vault-password-file", inputs["vault_password_file"]])

        response = self._run_ansible_cmd(*cmd)

        if response.success:
            return AdapterResponse.success_response(
                data={"file": file_path},
                status="encrypted"
            ).to_dict()

        return response.to_dict()

    async def _vault_decrypt(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt file with Ansible Vault

        Inputs:
            file: File to decrypt
            vault_password_file: Password file (optional)
        """
        file_path = inputs.get("file")
        if not file_path:
            return AdapterResponse.error_response(
                error="file is required"
            ).to_dict()

        cmd = ["ansible-vault", "decrypt", file_path]

        if "vault_password_file" in inputs:
            cmd.extend(["--vault-password-file", inputs["vault_password_file"]])

        response = self._run_ansible_cmd(*cmd)

        if response.success:
            return AdapterResponse.success_response(
                data={"file": file_path},
                status="decrypted"
            ).to_dict()

        return response.to_dict()
