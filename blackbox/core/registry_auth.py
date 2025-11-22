# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
Private Docker Registry Support
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional


class PrivateRegistryAuth:
    """
    Handles authentication for private Docker registries.

    Supports:
    - Docker Hub
    - GitHub Container Registry (ghcr.io)
    - Google Container Registry (gcr.io)
    - AWS ECR
    - Azure Container Registry
    - Self-hosted registries
    """

    def __init__(self):
        """Initialize registry auth."""
        self.logger = logging.getLogger("bbx.registry_auth")
        self.docker_config = Path.home() / ".docker" / "config.json"

    def login(
        self,
        registry: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ) -> bool:
        """
        Login to private registry.

        Args:
            registry: Registry URL (e.g., 'ghcr.io', 'gcr.io')
            username: Username
            password: Password or token
            token: Access token (alternative to username/password)

        Returns:
            True if login successful
        """
        import subprocess

        # Try token-based auth first
        if token:
            password = token
            username = username or "oauth2accesstoken"  # For GCR

        # Check env vars as fallback
        if not password:
            password = os.getenv(f"{registry.replace('.', '_').upper()}_TOKEN")

        if not username or not password:
            self.logger.error(f"Missing credentials for {registry}")
            return False

        try:
            # Use docker login
            result = subprocess.run(
                ["docker", "login", registry, "-u", username, "--password-stdin"],
                input=password,
                text=True,
                capture_output=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.logger.info(f"✅ Logged in to {registry}")
                return True
            else:
                self.logger.error(f"❌ Login failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Login error: {e}")
            return False

    def login_ecr(self, region: str = "us-east-1") -> bool:
        """Login to AWS ECR using AWS credentials."""
        import subprocess

        try:
            # Get ECR login token using AWS CLI
            result = subprocess.run(
                ["aws", "ecr", "get-login-password", "--region", region],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False

            password = result.stdout.strip()
            registry = f"{self._get_aws_account_id()}.dkr.ecr.{region}.amazonaws.com"

            return self.login(registry, username="AWS", password=password)

        except Exception as e:
            self.logger.error(f"ECR login failed: {e}")
            return False

    def login_gcr(self, project_id: str) -> bool:
        """Login to Google Container Registry."""
        import subprocess

        try:
            # Use gcloud for token
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False

            token = result.stdout.strip()
            return self.login("gcr.io", token=token)

        except Exception as e:
            self.logger.error(f"GCR login failed: {e}")
            return False

    def is_logged_in(self, registry: str) -> bool:
        """Check if already logged in to registry."""
        if not self.docker_config.exists():
            return False

        try:
            with open(self.docker_config, "r") as f:
                config = json.load(f)

            auths = config.get("auths", {})
            return registry in auths

        except Exception:
            return False

    def _get_aws_account_id(self) -> str:
        """Get AWS account ID."""
        import subprocess

        result = subprocess.run(
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--query",
                "Account",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return result.stdout.strip() if result.returncode == 0 else ""
