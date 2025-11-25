# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Docker Adapter

Provides Docker integration with:
- Container management (run, stop, remove)
- Image management (build, pull, push)
- Docker Compose integration
- Logs and inspection
- Cross-platform support

Examples:
    # Run a container
    - id: start_db
      mcp: bbx.docker
      method: run
      inputs:
        image: "postgres:15"
        name: "my_postgres"
        environment:
          POSTGRES_PASSWORD: "secret"
        ports:
          - "5432:5432"

    # Build an image
    - id: build_app
      mcp: bbx.docker
      method: build
      inputs:
        path: "./app"
        tag: "myapp:latest"

    # Docker Compose
    - id: compose_up
      mcp: bbx.docker
      method: compose_up
      inputs:
        file: "docker-compose.yml"
"""

import json
from typing import Any, Dict

from blackbox.core.base_adapter import AdapterResponse, CLIAdapter


class DockerAdapter(CLIAdapter):
    """BBX Adapter for Docker operations"""

    def __init__(self):
        super().__init__(
            adapter_name="Docker",
            cli_tool="docker",
            version_args=["--version"],
            required=True,
        )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Docker method"""
        self.log_execution(method, inputs)

        handlers = {
            "run": self._run_container,
            "stop": self._stop_container,
            "remove": self._remove_container,
            "build": self._build_image,
            "pull": self._pull_image,
            "push": self._push_image,
            "logs": self._get_logs,
            "inspect": self._inspect_container,
            "ps": self._list_containers,
            "compose_up": self._compose_up,
            "compose_down": self._compose_down,
        }

        handler = handlers.get(method)
        if not handler:
            raise ValueError(f"Unknown method: {method}")

        try:
            result = await handler(inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            return AdapterResponse.error_response(error=str(e)).to_dict()

    # Container Operations

    async def _run_container(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a Docker container

        Inputs:
            image: Image name
            name: Container name (optional)
            ports: Port mappings (optional)
            environment: Environment variables (optional)
            volumes: Volume mounts (optional)
            command: Command to run (optional)
            detach: Run in background (default: True)
        """
        image = inputs.get("image")
        if not image:
            return AdapterResponse.error_response(error="image is required").to_dict()

        args = ["run"]

        # Detach by default
        if inputs.get("detach", True):
            args.append("-d")

        # Container name
        if "name" in inputs:
            args.extend(["--name", inputs["name"]])

        # Port mappings
        for port in inputs.get("ports", []):
            args.extend(["-p", port])

        # Environment variables
        for key, value in inputs.get("environment", {}).items():
            args.extend(["-e", f"{key}={value}"])

        # Volume mounts
        for volume in inputs.get("volumes", []):
            args.extend(["-v", volume])

        # Image
        args.append(image)

        # Command
        if "command" in inputs:
            args.extend(inputs["command"].split())

        response = self.run_command(*args)

        if response.success:
            container_id = (
                response.data.strip()
                if isinstance(response.data, str)
                else str(response.data)
            )
            return AdapterResponse.success_response(
                data={
                    "container_id": container_id,
                    "name": inputs.get("name"),
                    "image": image,
                },
                status="running",
            ).to_dict()

        return response.to_dict()

    async def _stop_container(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a container"""
        container = inputs.get("name") or inputs.get("id")
        if not container:
            return AdapterResponse.error_response(
                error="name or id is required"
            ).to_dict()

        response = self.run_command("stop", container)

        if response.success:
            return AdapterResponse.success_response(
                data={"container": container}, status="stopped"
            ).to_dict()

        return response.to_dict()

    async def _remove_container(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a container"""
        container = inputs.get("name") or inputs.get("id")
        if not container:
            return AdapterResponse.error_response(
                error="name or id is required"
            ).to_dict()

        args = ["rm"]

        if inputs.get("force"):
            args.append("-f")

        args.append(container)
        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data={"container": container}, status="removed"
            ).to_dict()

        return response.to_dict()

    # Image Operations

    async def _build_image(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a Docker image

        Inputs:
            path: Build context path
            tag: Image tag
            dockerfile: Dockerfile path (optional)
            build_args: Build arguments (optional)
        """
        path = inputs.get("path")
        tag = inputs.get("tag")

        if not all([path, tag]):
            return AdapterResponse.error_response(
                error="path and tag are required"
            ).to_dict()

        args = ["build", "-t", tag]

        # Dockerfile
        if "dockerfile" in inputs:
            args.extend(["-f", inputs["dockerfile"]])

        # Build arguments
        for key, value in inputs.get("build_args", {}).items():
            args.extend(["--build-arg", f"{key}={value}"])

        args.append(path)

        response = self.run_command(*args, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"tag": tag, "path": path}, status="built"
            ).to_dict()

        return response.to_dict()

    async def _pull_image(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pull a Docker image"""
        image = inputs.get("image")
        if not image:
            return AdapterResponse.error_response(error="image is required").to_dict()

        response = self.run_command("pull", image, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"image": image}, status="pulled"
            ).to_dict()

        return response.to_dict()

    async def _push_image(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Push a Docker image"""
        image = inputs.get("image")
        if not image:
            return AdapterResponse.error_response(error="image is required").to_dict()

        response = self.run_command("push", image, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"image": image}, status="pushed"
            ).to_dict()

        return response.to_dict()

    # Inspection and Logs

    async def _get_logs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get container logs"""
        container = inputs.get("name") or inputs.get("id")
        if not container:
            return AdapterResponse.error_response(
                error="name or id is required"
            ).to_dict()

        args = ["logs"]

        if inputs.get("follow"):
            args.append("-f")

        if "tail" in inputs:
            args.extend(["--tail", str(inputs["tail"])])

        args.append(container)
        response = self.run_command(*args)

        if response.success:
            logs = (
                response.data if isinstance(response.data, str) else str(response.data)
            )
            return AdapterResponse.success_response(
                data={"container": container, "logs": logs}, status="ok"
            ).to_dict()

        return response.to_dict()

    async def _inspect_container(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a container"""
        container = inputs.get("name") or inputs.get("id")
        if not container:
            return AdapterResponse.error_response(
                error="name or id is required"
            ).to_dict()

        response = self.run_command("inspect", container)

        if response.success:
            try:
                data_str = (
                    response.data
                    if isinstance(response.data, str)
                    else str(response.data)
                )
                data = json.loads(data_str)
                return AdapterResponse.success_response(
                    data={"container": container, "info": data[0] if data else {}},
                    status="ok",
                ).to_dict()
            except json.JSONDecodeError:
                return AdapterResponse.error_response(
                    error="Failed to parse inspect output"
                ).to_dict()

        return response.to_dict()

    async def _list_containers(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List containers"""
        args = ["ps"]

        if inputs.get("all"):
            args.append("-a")

        args.extend(["--format", "{{json .}}"])

        response = self.run_command(*args)

        if response.success:
            containers = []
            data_str = (
                response.data if isinstance(response.data, str) else str(response.data)
            )
            for line in data_str.split("\n"):
                if line.strip():
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            return AdapterResponse.success_response(
                data={"containers": containers, "count": len(containers)}, status="ok"
            ).to_dict()

        return response.to_dict()

    # Docker Compose Operations

    async def _compose_up(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Docker Compose up"""
        compose_file = inputs.get("file", "docker-compose.yml")
        args = ["compose", "-f", compose_file, "up"]

        if inputs.get("detach", True):
            args.append("-d")

        if inputs.get("build"):
            args.append("--build")

        response = self.run_command(*args, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"file": compose_file}, status="up"
            ).to_dict()

        return response.to_dict()

    async def _compose_down(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Docker Compose down"""
        compose_file = inputs.get("file", "docker-compose.yml")
        args = ["compose", "-f", compose_file, "down"]

        if inputs.get("volumes"):
            args.append("-v")

        response = self.run_command(*args)

        if response.success:
            return AdapterResponse.success_response(
                data={"file": compose_file}, status="down"
            ).to_dict()

        return response.to_dict()
