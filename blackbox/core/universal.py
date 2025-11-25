# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Universal Adapter
The "One Adapter to Rule Them All".
Driven by YAML definitions and Jinja2 templating.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import jinja2

from blackbox.core.auth import AuthRegistry
from blackbox.core.base_adapter import AdapterResponse, DockerizedAdapter


class UniversalAdapter(DockerizedAdapter):
    """
    Generic adapter that executes Docker commands based on a YAML definition.
    """

    def __init__(self, definition: Optional[Dict[str, Any]] = None):
        """
        Initialize with a definition dict.

        Args:
            definition: The parsed YAML definition containing:
                - id: Adapter ID
                - uses: Docker image
                - auth: Auth config
                - cmd: Command template list
                - output_parser: Output parsing config
        """
        self.definition = definition or {}
        self.logger = logging.getLogger(
            f"bbx.universal.{self.definition.get('id', 'unknown')}"
        )

        # Parse Docker image from 'uses' (e.g., "docker://image:tag")
        image_str = self.definition.get(
            "uses", "alpine:latest"
        )  # Default to alpine if empty
        if image_str.startswith("docker://"):
            image = image_str.replace("docker://", "")
        else:
            image = image_str

        super().__init__(
            adapter_name=self.definition.get("id", "Universal"),
            docker_image=image,
            cli_tool="docker",  # We use docker directly
            required=True,
        )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute the universal adapter logic.

        Supports two modes:
        1. Definition-based: Configuration provided in __init__ (YAML library)
        2. Dynamic: Configuration provided in inputs (Ad-hoc usage)
        """
        self.log_execution(method, inputs)

        try:
            # Merge definition and inputs. Inputs override definition.
            # This allows "Dynamic Mode" where 'uses', 'cmd', etc. are passed as inputs.
            config = self.definition.copy()
            config.update(inputs)

            # 1. Resolve Image
            image_str = config.get("uses")
            if not image_str:
                # If not in config, check if we initialized with one (from super)
                if self.docker_image and self.docker_image != "alpine:latest":
                    image_str = self.docker_image
                else:
                    raise ValueError(
                        "No Docker image specified (missing 'uses' in inputs or definition)"
                    )

            if image_str.startswith("docker://"):
                image = image_str.replace("docker://", "")
            else:
                image = image_str

            # Update the base adapter's image for this execution
            self.docker_image = image

            # 2. Prepare Context for Templating
            context = {
                "inputs": inputs,
                "args": inputs,  # Alias
                "env": os.environ.copy(),  # Inject host env
            }

            # 3. Render Command Template
            cmd_template = config.get("cmd", [])
            if isinstance(cmd_template, str):
                cmd_template = [cmd_template]  # Handle string command

            rendered_cmd = self._render_command(cmd_template, context)

            # 4. Inject Authentication
            env = {}
            volumes = {}
            auth_config = config.get("auth")
            if auth_config:
                provider_name = auth_config.get("type")
                provider = AuthRegistry.get_provider(provider_name)
                if provider:
                    auth_env, auth_volumes = provider.inject(auth_config)
                    env.update(auth_env)
                    volumes.update(auth_volumes)
                else:
                    self.logger.warning(f"Unknown auth provider: {provider_name}")

            # 5. Handle Working Directory
            working_dir = config.get("working_dir")

            # 6. Execute Docker Command
            response = self.run_command(
                *rendered_cmd, env=env, volumes=volumes, working_dir=working_dir
            )

            if not response.success:
                return response.to_dict()

            # 7. Parse Output
            parsed_data = self._parse_output(response.data, config.get("output_parser"))

            return AdapterResponse.success_response(
                data=parsed_data, status="success"
            ).to_dict()

        except Exception as e:
            self.logger.exception("Universal execution failed")
            return AdapterResponse.error_response(error=str(e)).to_dict()

    def _render_command(
        self, template: List[str], context: Dict[str, Any]
    ) -> List[str]:
        """Render the command list using Jinja2"""
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        rendered = []

        for item in template:
            # Handle conditional blocks in Jinja (tricky in a list)
            # Strategy: Render the item. If it's empty, skip it.
            # Better Strategy: The template might be a single string or list.
            # If it's a list of strings, we render each string.

            # Special case: If item contains {% if %}, it might evaluate to empty string
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
        """Parse raw output based on config"""
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

                # Support JMESPath query if needed (optional enhancement)
                query = parser_config.get("query")
                if query:
                    import jmespath  # type: ignore

                    return jmespath.search(query, data)

                return data
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse output as JSON")
                return raw_output

        return raw_output
