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
BBX Core - Unit Tests for Auth Providers
Bottom-up testing: Layer 1 - No external dependencies
"""

from unittest.mock import patch
from blackbox.core.auth import (
    KubeConfigProvider,
    AWSCredentialsProvider,
    AuthRegistry
)


class TestKubeConfigProvider:
    """Test Kubernetes config provider."""
    
    def test_inject_when_kubeconfig_exists(self, tmp_path):
        """Test successful injection when kubeconfig exists."""
        kube_dir = tmp_path / ".kube"
        kube_dir.mkdir()
        kubeconfig = kube_dir / "config"
        kubeconfig.write_text("fake config")
        
        with patch("pathlib.Path.home", return_value=tmp_path):
            provider = KubeConfigProvider()
            env, volumes = provider.inject({})
            
            assert env["KUBECONFIG"] == "/root/.kube/config"
            assert str(kubeconfig) in volumes
    
    def test_inject_when_kubeconfig_missing(self, tmp_path):
        """Test injection when kubeconfig doesn't exist."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            provider = KubeConfigProvider()
            env, volumes = provider.inject({})
            
            assert env == {}
            assert volumes == {}


class TestAWSCredentialsProvider:
    """Test AWS credentials provider."""
    
    def test_inject_env_vars(self, monkeypatch, tmp_path):
        """Test AWS env var injection."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake_secret")
        
        with patch("pathlib.Path.home", return_value=tmp_path):
            provider = AWSCredentialsProvider()
            env, volumes = provider.inject({})
            
            assert env["AWS_ACCESS_KEY_ID"] == "fake_key"
            assert env["AWS_SECRET_ACCESS_KEY"] == "fake_secret"


class TestAuthRegistry:
    """Test Auth Registry."""
    
    def test_get_kubeconfig_provider(self):
        """Test getting kubeconfig provider."""
        provider = AuthRegistry.get_provider("kubeconfig")
        assert isinstance(provider, KubeConfigProvider)
    
    def test_get_unknown_provider(self):
        """Test getting unknown provider returns None."""
        provider = AuthRegistry.get_provider("unknown")
        assert provider is None
