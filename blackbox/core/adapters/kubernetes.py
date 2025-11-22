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
BBX Kubernetes Adapter

Provides full Kubernetes orchestration:
- Resource management (create/apply/delete)
- Deployment operations
- Service exposure
- Pod management
- Namespace operations
- Helm chart deployment
- kubectl command execution

Examples:
    # Apply manifest
    - id: deploy_app
      mcp: bbx.k8s
      method: apply
      inputs:
        file: "deployment.yaml"
        namespace: "production"

    # Scale deployment
    - id: scale_up
      mcp: bbx.k8s
      method: scale
      inputs:
        deployment: "my-app"
        replicas: 5
        namespace: "production"

    # Get pods
    - id: list_pods
      mcp: bbx.k8s
      method: get
      inputs:
        resource: "pods"
        namespace: "production"
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse


class KubernetesAdapter(DockerizedAdapter):
    """BBX Adapter for Kubernetes operations (Dockerized)"""

    def __init__(self):
        super().__init__(
            adapter_name="Kubernetes",
            docker_image="dtzar/helm-kubectl:latest",
            cli_tool="kubectl",
            version_args=["version", "--client"],
            required=True
        )

    def run_command(self, *args, **kwargs):
        """Override run_command to handle kubectl/helm and auth"""
        # Inject Kubeconfig
        env = kwargs.get("env", {}) or {}
        volumes = kwargs.get("volumes", {}) or {}
        
        # Mount ~/.kube if it exists
        home = Path.home()
        kube_dir = home / ".kube"
        if kube_dir.exists():
            # Mount to /root/.kube (container runs as root usually)
            # or /home/argocd/.kube depending on image user.
            # dtzar/helm-kubectl runs as root? Let's assume root for now.
            volumes[str(kube_dir)] = "/root/.kube"
            
        # Pass KUBECONFIG env var if set
        if os.environ.get("KUBECONFIG"):
            env["KUBECONFIG"] = os.environ["KUBECONFIG"]
            # If KUBECONFIG points to a file, we might need to mount it if not in ~/.kube
            # For simplicity, assume standard ~/.kube setup or user handles mounts via env
            
        kwargs["env"] = env
        kwargs["volumes"] = volumes
        
        # Handle command prefixing
        # If args[0] is not helm or kubectl, prepend kubectl
        # This allows self.run_command("apply", ...) to work as "kubectl apply ..."
        args_list = list(args)
        if args_list:
            cmd = args_list[0]
            if cmd not in ["kubectl", "helm"]:
                args_list.insert(0, "kubectl")
                
        return super().run_command(*args_list, **kwargs)

    def _run_helm_command(self, *args, timeout=300):
        """Run helm command using Dockerized execution"""
        # args usually start with subcommand like "install", "upgrade"
        # We prepend "helm" and call run_command
        return self.run_command("helm", *args, timeout=timeout)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Kubernetes method"""
        self.log_execution(method, inputs)

        handlers = {
            "apply": self._apply,
            "delete": self._delete,
            "get": self._get,
            "describe": self._describe,
            "scale": self._scale,
            "rollout": self._rollout,
            "logs": self._logs,
            "exec": self._exec,
            "port_forward": self._port_forward,
            "create_namespace": self._create_namespace,
            "helm_install": self._helm_install,
            "helm_upgrade": self._helm_upgrade,
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

    # Resource Management

    async def _apply(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Kubernetes manifest

        Inputs:
            file: Path to manifest file or "-" for stdin
            content: YAML content (if file is "-")
            namespace: Namespace (optional)
            dry_run: Dry run mode (optional)
        """
        args = ["apply"]

        # Namespace
        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        # Dry run
        if inputs.get("dry_run"):
            args.append("--dry-run=client")

        # File or content
        if "file" in inputs and inputs["file"] != "-":
            args.extend(["-f", inputs["file"]])
        else:
            args.extend(["-f", "-"])

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"file": inputs.get("file")},
                status="applied"
            ).to_dict()

        return response.to_dict()

    async def _delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete Kubernetes resources

        Inputs:
            file: Path to manifest (optional)
            resource: Resource type (e.g., "deployment")
            name: Resource name
            namespace: Namespace (optional)
        """
        args = ["delete"]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        if "file" in inputs:
            args.extend(["-f", inputs["file"]])
        elif "resource" in inputs and "name" in inputs:
            args.extend([inputs["resource"], inputs["name"]])
        else:
            return AdapterResponse.error_response(
                error="Either file or (resource and name) are required"
            ).to_dict()

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"resource": inputs.get("resource"), "name": inputs.get("name")},
                status="deleted"
            ).to_dict()

        return response.to_dict()

    async def _get(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Kubernetes resources

        Inputs:
            resource: Resource type (e.g., "pods", "services")
            name: Specific resource name (optional)
            namespace: Namespace (optional)
            all_namespaces: Search all namespaces (optional)
            output: Output format (json/yaml/wide) (default: json)
        """
        resource = inputs.get("resource")
        if not resource:
            return AdapterResponse.error_response(
                error="resource is required"
            ).to_dict()

        args = ["get", resource]

        if "name" in inputs:
            args.append(inputs["name"])

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        if inputs.get("all_namespaces"):
            args.append("--all-namespaces")

        # Output format
        output_format = inputs.get("output", "json")
        args.extend(["-o", output_format])

        response = self.run_command(*args, output_format=None)

        if response.success and output_format == "json":
            try:
                data_str = response.data if isinstance(response.data, str) else str(response.data)
                data = json.loads(data_str)
                return AdapterResponse.success_response(
                    data={
                        "resource": resource,
                        "data": data,
                        "items": data.get("items", [data]) if "items" in data else [data]
                    },
                    status="ok"
                ).to_dict()
            except json.JSONDecodeError:
                pass

        if response.success:
            return AdapterResponse.success_response(
                data={"resource": resource, "output": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    async def _describe(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Describe Kubernetes resource

        Inputs:
            resource: Resource type
            name: Resource name
            namespace: Namespace (optional)
        """
        resource = inputs.get("resource")
        name = inputs.get("name")

        if not all([resource, name]):
            return AdapterResponse.error_response(
                error="resource and name are required"
            ).to_dict()

        args = ["describe", resource, name]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"resource": resource, "name": name, "description": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Deployment Operations

    async def _scale(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale deployment/replicaset

        Inputs:
            deployment: Deployment name
            replicas: Number of replicas
            namespace: Namespace (optional)
        """
        deployment = inputs.get("deployment")
        replicas = inputs.get("replicas")

        if not all([deployment, replicas is not None]):
            return AdapterResponse.error_response(
                error="deployment and replicas are required"
            ).to_dict()

        args = ["scale", f"deployment/{deployment}", f"--replicas={replicas}"]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"deployment": deployment, "replicas": replicas},
                status="scaled"
            ).to_dict()

        return response.to_dict()

    async def _rollout(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage rollout

        Inputs:
            action: Action (restart/status/undo/history)
            resource: Resource (e.g., "deployment/my-app")
            namespace: Namespace (optional)
        """
        action = inputs.get("action")
        resource = inputs.get("resource")

        if not all([action, resource]):
            return AdapterResponse.error_response(
                error="action and resource are required"
            ).to_dict()

        args = ["rollout", action, resource]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"action": action, "resource": resource},
                status="ok"
            ).to_dict()

        return response.to_dict()

    # Pod Operations

    async def _logs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get pod logs

        Inputs:
            pod: Pod name
            container: Container name (optional)
            namespace: Namespace (optional)
            follow: Follow logs (optional)
            tail: Number of lines (optional)
        """
        pod = inputs.get("pod")
        if not pod:
            return AdapterResponse.error_response(
                error="pod is required"
            ).to_dict()

        args = ["logs", pod]

        if "container" in inputs:
            args.extend(["-c", inputs["container"]])

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        if inputs.get("follow"):
            args.append("-f")

        if "tail" in inputs:
            args.extend(["--tail", str(inputs["tail"])])

        response = self.run_command(*args, timeout=60, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"pod": pod, "logs": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    async def _exec(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute command in pod

        Inputs:
            pod: Pod name
            command: Command to execute
            container: Container name (optional)
            namespace: Namespace (optional)
        """
        pod = inputs.get("pod")
        command = inputs.get("command")

        if not all([pod, command]):
            return AdapterResponse.error_response(
                error="pod and command are required"
            ).to_dict()

        args = ["exec", pod]

        if "container" in inputs:
            args.extend(["-c", inputs["container"]])

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        args.append("--")

        # Add command
        if isinstance(command, str):
            args.extend(command.split())
        else:
            args.extend(command)

        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"pod": pod, "output": response.data},
                status="ok"
            ).to_dict()

        return response.to_dict()

    async def _port_forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Port forward to pod/service

        Inputs:
            resource: Resource (e.g., "pod/my-pod" or "service/my-svc")
            ports: Port mapping (e.g., "8080:80")
            namespace: Namespace (optional)
        """
        resource = inputs.get("resource")
        ports = inputs.get("ports")

        if not all([resource, ports]):
            return AdapterResponse.error_response(
                error="resource and ports are required"
            ).to_dict()

        args = ["port-forward", resource, ports]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        # Note: This runs in background
        response = self.run_command(*args, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"resource": resource, "ports": ports},
                status="forwarding"
            ).to_dict()

        return response.to_dict()

    # Namespace Operations

    async def _create_namespace(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create namespace"""
        namespace = inputs.get("namespace")
        if not namespace:
            return AdapterResponse.error_response(
                error="namespace is required"
            ).to_dict()

        response = self.run_command("create", "namespace", namespace, output_format=None)

        if response.success:
            return AdapterResponse.success_response(
                data={"namespace": namespace},
                status="created"
            ).to_dict()

        return response.to_dict()

    # Helm Operations

    async def _helm_install(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install Helm chart

        Inputs:
            release: Release name
            chart: Chart name
            namespace: Namespace
            values: Values dict (optional)
            values_file: Values file (optional)
        """
        release = inputs.get("release")
        chart = inputs.get("chart")

        if not all([release, chart]):
            return AdapterResponse.error_response(
                error="release and chart are required"
            ).to_dict()

        args = ["install", release, chart]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"], "--create-namespace"])

        if "values_file" in inputs:
            args.extend(["-f", inputs["values_file"]])

        if "values" in inputs:
            for key, value in inputs["values"].items():
                args.extend(["--set", f"{key}={value}"])

        response = self._run_helm_command(*args, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"release": release, "chart": chart},
                status="installed"
            ).to_dict()

        return response.to_dict()

    async def _helm_upgrade(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade Helm release"""
        release = inputs.get("release")
        chart = inputs.get("chart")

        if not all([release, chart]):
            return AdapterResponse.error_response(
                error="release and chart are required"
            ).to_dict()

        args = ["upgrade", release, chart]

        if "namespace" in inputs:
            args.extend(["-n", inputs["namespace"]])

        if inputs.get("install"):
            args.append("--install")

        if "values_file" in inputs:
            args.extend(["-f", inputs["values_file"]])

        if "values" in inputs:
            for key, value in inputs["values"].items():
                args.extend(["--set", f"{key}={value}"])

        response = self._run_helm_command(*args, timeout=600)

        if response.success:
            return AdapterResponse.success_response(
                data={"release": release},
                status="upgraded"
            ).to_dict()

        return response.to_dict()
