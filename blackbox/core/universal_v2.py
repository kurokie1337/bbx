# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
BBX Universal Adapter - Enhanced with streaming and progress
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import jinja2

from blackbox.core.auth import AuthRegistry
from blackbox.core.base_adapter import (AdapterErrorType, AdapterResponse,
                                        DockerizedAdapter)


class UniversalAdapterV2(DockerizedAdapter):
    """
    Enhanced Universal Adapter with:
    - Real-time output streaming
    - Progress indicators
    - Multi-step command support
    - Resource limits
    """

    def __init__(self, definition: Optional[Dict[str, Any]] = None):
        """Initialize with optional definition."""
        self.definition = definition or {}
        self.logger = logging.getLogger(
            f"bbx.universal.{self.definition.get('id', 'unknown')}"
        )

        # Parse Docker image
        image_str = self.definition.get("uses", "alpine:latest")
        if image_str.startswith("docker://"):
            image = image_str.replace("docker://", "")
        else:
            image = image_str

        super().__init__(
            adapter_name=self.definition.get("id", "Universal"),
            docker_image=image,
            cli_tool="docker",
            required=True,
        )

        # Enhanced features
        self.enable_streaming = True
        self.show_progress = True

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute with enhanced features."""
        self.log_execution(method, inputs)

        try:
            # Merge config
            config = self.definition.copy()
            config.update(inputs)

            # Multi-step support
            if "steps" in config:
                return await self._execute_multi_step(config["steps"], inputs)

            # Single command execution
            return await self._execute_single(config, inputs)

        except Exception as e:
            self.logger.exception("Universal execution failed")
            return AdapterResponse.error_response(error=str(e)).to_dict()

    async def _execute_single(
        self, config: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single command."""
        # Image resolution
        image_str = config.get("uses")
        if not image_str:
            if self.docker_image and self.docker_image != "alpine:latest":
                image_str = self.docker_image
            else:
                raise ValueError("No Docker image specified")

        if image_str.startswith("docker://"):
            image = image_str.replace("docker://", "")
        else:
            image = image_str

        self.docker_image = image

        # Context for templating
        context = {
            "inputs": inputs,
            "args": inputs,
            "env": os.environ.copy(),
        }

        # Render command
        cmd_template = config.get("cmd", [])
        if isinstance(cmd_template, str):
            cmd_template = [cmd_template]

        rendered_cmd = self._render_command(cmd_template, context)

        # Render environment variables
        env = {}
        if "env" in config:
            for key, value in config["env"].items():
                if isinstance(value, str):
                    rendered_value = jinja2.Template(value).render(**context)
                    env[key] = rendered_value
                else:
                    env[key] = str(value)

        # Auth injection
        volumes = {}
        auth_config = config.get("auth")
        if auth_config:
            provider_name = auth_config.get("type")
            provider = AuthRegistry.get_provider(provider_name)
            if provider:
                auth_env, auth_volumes = provider.inject(auth_config)
                env.update(auth_env)
                volumes.update(auth_volumes)

        # Add volumes from config
        if "volumes" in config:
            volumes.update(config["volumes"])

        # Resource limits and timeout
        resource_limits = config.get("resources", {})
        timeout = config.get("timeout")  # Timeout in seconds

        # Execute with streaming
        response = await self._run_with_streaming(
            rendered_cmd,
            env=env,
            volumes=volumes,
            working_dir=config.get("working_dir"),
            resources=resource_limits,
            timeout=timeout,
        )

        if not response.success:
            return response.to_dict()

        # Parse output
        parsed_data = self._parse_output(response.data, config.get("output_parser"))

        return AdapterResponse.success_response(
            data=parsed_data, status="success"
        ).to_dict()

    async def _execute_multi_step(
        self, steps: List[Dict], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multiple steps sequentially."""
        results: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {"inputs": inputs, "results": []}

        for i, step in enumerate(steps):
            self.logger.info(
                f"Executing step {i+1}/{len(steps)}: {step.get('name', 'unnamed')}"
            )

            # Merge previous results into context
            step_context = context.copy()
            step_context.update(step.get("inputs", {}))

            result = await self._execute_single(step, step_context)
            results.append(result)

            # Break on error if not continue_on_error
            if not result.get("success") and not step.get("continue_on_error", False):
                break

            # Add to context for next step
            context["results"].append(result.get("data"))

        return {
            "success": all(r.get("success") for r in results),
            "steps": results,
            "data": results[-1].get("data") if results else None,
        }

    async def _run_with_streaming(
        self,
        cmd: List[str],
        env: Dict[str, str],
        volumes: Dict[str, str],
        working_dir: Optional[str],
        resources: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> AdapterResponse:
        """Run command with real-time output streaming and optional timeout."""
        import subprocess
        from pathlib import Path

        # Build docker command
        docker_cmd = ["run", "--rm"]

        # Ensure docker_image is set
        if not self.docker_image:
            return AdapterResponse.error_response(
                error="Docker image not specified",
                error_type=AdapterErrorType.EXECUTION_ERROR,
            )

        # Show progress
        if self.show_progress and not self._image_exists(self.docker_image):
            print(f"📦 Pulling {self.docker_image}...", flush=True)
            # Pull with progress
            pull_result = subprocess.run(
                ["docker", "pull", self.docker_image],
                capture_output=False,  # Show progress to stdout
            )
            if pull_result.returncode != 0:
                return AdapterResponse.error_response(
                    error=f"Failed to pull image {self.docker_image}",
                    error_type=AdapterErrorType.EXECUTION_ERROR,
                )

        # Resource limits
        if resources.get("cpu"):
            docker_cmd.extend(["--cpus", str(resources["cpu"])])
        if resources.get("memory"):
            docker_cmd.extend(["--memory", resources["memory"]])

        # Volumes
        host_cwd = Path.cwd()
        docker_cmd.extend(["-v", f"{host_cwd}:/workspace"])

        # Working directory
        if working_dir:
            docker_cmd.extend(["-w", working_dir])
        else:
            docker_cmd.extend(["-w", "/workspace"])

        for host_path, container_path in volumes.items():
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Environment
        for key, value in env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # UID/GID for Linux/Mac
        import platform

        if platform.system() != "Windows":
            if hasattr(os, "getuid") and hasattr(os, "getgid"):
                try:
                    uid = os.getuid()  # type: ignore
                    gid = os.getgid()  # type: ignore
                    docker_cmd.extend(["--user", f"{uid}:{gid}"])
                except AttributeError:
                    pass

        # Image and command
        docker_cmd.append(self.docker_image)
        docker_cmd.extend(cmd)

        self.logger.debug(f"Docker command: {' '.join(docker_cmd)}")

        # Execute with streaming
        if self.enable_streaming:
            print(f"▶️  Running: {' '.join(cmd[:3])}...", flush=True)

        # Execute with timeout
        try:
            result = subprocess.run(
                ["docker"] + docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,  # Will raise TimeoutExpired if exceeds
            )

            stdout = result.stdout
            stderr = result.stderr

            if self.enable_streaming and stdout:
                print(stdout, flush=True)
            if self.enable_streaming and stderr:
                print(f"⚠️  {stderr}", file=sys.stderr, flush=True)

            if result.returncode == 0:
                return AdapterResponse.success_response(
                    data=stdout,
                    exit_code=result.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            else:
                return AdapterResponse.error_response(
                    error=stderr
                    or f"Command failed with exit code {result.returncode}",
                    error_type=AdapterErrorType.EXECUTION_ERROR,
                    exit_code=result.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
        except subprocess.TimeoutExpired:
            return AdapterResponse.error_response(
                error=f"Command timed out after {timeout} seconds",
                error_type=AdapterErrorType.TIMEOUT_ERROR,
            )

    def _render_command(
        self, template: List[str], context: Dict[str, Any]
    ) -> List[str]:
        """Render command with Jinja2."""
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        rendered = []

        for item in template:
            try:
                tmpl = env.from_string(item)
                result = tmpl.render(**context)
                if result.strip():
                    rendered.append(result)
            except jinja2.UndefinedError as e:
                raise ValueError(f"Missing input for template '{item}': {e}")

        return rendered

    def _parse_output(
        self, raw_output: Any, parser_config: Optional[Dict[str, Any]]
    ) -> Any:
        """Parse output."""
        if not parser_config:
            return raw_output

        parser_type = parser_config.get("type", "text")

        if parser_type == "json":
            try:
                data = (
                    json.loads(raw_output)
                    if isinstance(raw_output, str)
                    else raw_output
                )

                query = parser_config.get("query")
                if query:
                    try:
                        import jmespath  # type: ignore

                        return jmespath.search(query, data)
                    except ImportError:
                        self.logger.warning("jmespath not installed, skipping query")
                        return data

                return data
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse output as JSON")
                return raw_output

        return raw_output

    def _image_exists(self, image: str) -> bool:
        """Check if Docker image exists locally."""
        import subprocess

        try:
            result = subprocess.run(
                ["docker", "images", "-q", image],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
