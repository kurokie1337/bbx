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
BBX Auth Providers
Kernel-level authentication injection for Universal Adapters.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class AuthProvider(ABC):
    """
    Abstract base class for Auth Providers.
    Responsible for injecting credentials into Docker containers.
    """

    @abstractmethod
    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Inject credentials.

        Args:
            config: Configuration dict from the adapter YAML (e.g., {"profile": "default"})

        Returns:
            Tuple of (env_vars, volume_mounts)
        """


class KubeConfigProvider(AuthProvider):
    """Provider for Kubernetes Config (~/.kube/config)"""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        # Default to standard location
        kube_config_path = Path.home() / ".kube" / "config"

        # Allow override via env var
        if os.environ.get("KUBECONFIG"):
            kube_config_path = Path(os.environ["KUBECONFIG"])

        if kube_config_path.exists():
            # Mount to standard location in container (root user usually)
            # Many images look at ~/.kube/config or /root/.kube/config
            volumes[str(kube_config_path)] = "/root/.kube/config"
            # Also set env var just in case
            env["KUBECONFIG"] = "/root/.kube/config"

        return env, volumes


class AWSCredentialsProvider(AuthProvider):
    """Provider for AWS Credentials (env vars or ~/.aws)"""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        # 1. Pass Environment Variables
        aws_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_PROFILE",
        ]

        for var in aws_vars:
            if os.environ.get(var):
                env[var] = os.environ[var]

        # 2. Mount ~/.aws directory if it exists
        aws_dir = Path.home() / ".aws"
        if aws_dir.exists():
            volumes[str(aws_dir)] = "/root/.aws"

        return env, volumes


class GCPCredentialsProvider(AuthProvider):
    """Provider for GCP Credentials (GOOGLE_APPLICATION_CREDENTIALS)"""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        # Pass Project ID
        if os.environ.get("CLOUDSDK_CORE_PROJECT"):
            env["CLOUDSDK_CORE_PROJECT"] = os.environ["CLOUDSDK_CORE_PROJECT"]

        # Handle Credentials File
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            host_path = Path(creds_path).resolve()
            if host_path.exists():
                container_path = "/tmp/gcp_credentials.json"
                volumes[str(host_path)] = container_path
                env["GOOGLE_APPLICATION_CREDENTIALS"] = container_path

        return env, volumes


class AuthRegistry:
    """Registry for Auth Providers"""

    _providers = {
        "kubeconfig": KubeConfigProvider(),
        "aws_credentials": AWSCredentialsProvider(),
        "gcp_credentials": GCPCredentialsProvider(),
    }

    @classmethod
    def get_provider(cls, name: str) -> Optional[AuthProvider]:
        return cls._providers.get(name)
