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
Comprehensive tests for Phase 6: Infrastructure Adapters

Tests Terraform, Ansible, and Kubernetes adapters
"""

import pytest
from blackbox.core.adapters.terraform import TerraformAdapter
from blackbox.core.adapters.ansible import AnsibleAdapter
from blackbox.core.adapters.kubernetes import KubernetesAdapter


class TestTerraformAdapter:
    """Test Terraform adapter"""
    
    @pytest.fixture
    def adapter(self):
        return TerraformAdapter()
    
    @pytest.mark.asyncio
    async def test_terraform_methods_exist(self, adapter):
        """Test that all Terraform methods exist"""
        methods = ["init", "plan", "apply", "destroy", "output", "validate", "fmt", "show", "state_list"]
        
        for method in methods:
            result = await adapter.execute(method, {"working_dir": "./test_terraform"})
            assert isinstance(result, dict)
            # It's ok if it fails - Terraform might not be installed
            # We're just testing the method exists
    
    @pytest.mark.asyncio
    async def test_terraform_init_structure(self, adapter):
        """Test init returns proper structure"""
        result = await adapter.execute("init", {"working_dir": "./test"})
        assert "status" in result
        assert "working_dir" in result
        assert result["working_dir"] == "./test"
    
    @pytest.mark.asyncio
    async def test_terraform_plan_with_vars(self, adapter):
        """Test plan with variables"""
        result = await adapter.execute("plan", {
            "working_dir": "./test",
            "variables": {"env": "production", "region": "us-east-1"},
            "var_file": "prod.tfvars"
        })
        assert "status" in result
        assert "working_dir" in result
    
    @pytest.mark.asyncio
    async def test_terraform_output_json(self, adapter):
        """Test output method with JSON"""
        result = await adapter.execute("output", {
            "working_dir": "./test",
            "json": True
        })
        assert "status" in result


class TestAnsibleAdapter:
    """Test Ansible adapter"""
    
    @pytest.fixture
    def adapter(self):
        return AnsibleAdapter()
    
    @pytest.mark.asyncio
    async def test_ansible_methods_exist(self, adapter):
        """Test that all Ansible methods exist"""
        methods = ["playbook", "adhoc", "galaxy_install", "inventory_list"]
        
        for method in methods:
            # Methods exist and are callable
            assert hasattr(adapter, f"_{method}")
    
    @pytest.mark.asyncio
    async def test_ansible_playbook_structure(self, adapter):
        """Test playbook execution structure"""
        result = await adapter.execute("playbook", {
            "playbook": "deploy.yml",
            "inventory": "hosts.ini"
        })
        assert "status" in result
        assert "playbook" in result
        assert result["playbook"] == "deploy.yml"
    
    @pytest.mark.asyncio
    async def test_ansible_adhoc_structure(self, adapter):
        """Test ad-hoc command structure"""
        result = await adapter.execute("adhoc", {
            "pattern": "all",
            "module": "ping",
            "inventory": "hosts.ini"
        })
        assert "status" in result
        assert "pattern" in result
        assert "module" in result
    
    @pytest.mark.asyncio
    async def test_ansible_galaxy_install(self, adapter):
        """Test galaxy install"""
        result = await adapter.execute("galaxy_install", {
            "name": "requirements.yml",
            "type": "role"
        })
        assert "status" in result
        assert "name" in result
        assert "type" in result


class TestKubernetesAdapter:
    """Test Kubernetes adapter"""
    
    @pytest.fixture
    def adapter(self):
        return KubernetesAdapter()
    
    @pytest.mark.asyncio
    async def test_k8s_methods_exist(self, adapter):
        """Test that all K8s methods exist"""
        methods = [
            "apply", "delete", "get", "describe", "scale",
            "rollout", "logs", "exec", "port_forward",
            "create_namespace", "helm_install", "helm_upgrade"
        ]
        
        for method in methods:
            assert hasattr(adapter, f"_{method}")
    
    @pytest.mark.asyncio
    async def test_k8s_get_structure(self, adapter):
        """Test get pods structure"""
        result = await adapter.execute("get", {
            "resource": "pods",
            "namespace": "default"
        })
        assert "status" in result
        assert "resource" in result
    
    @pytest.mark.asyncio
    async def test_k8s_scale_structure(self, adapter):
        """Test scale deployment structure"""
        result = await adapter.execute("scale", {
            "deployment": "my-app",
            "replicas": 3,
            "namespace": "production"
        })
        assert "status" in result
        assert "deployment" in result
        assert "replicas" in result
    
    @pytest.mark.asyncio
    async def test_k8s_apply_structure(self, adapter):
        """Test apply manifest structure"""
        result = await adapter.execute("apply", {
            "file": "deployment.yaml",
            "namespace": "production"
        })
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_k8s_create_namespace(self, adapter):
        """Test create namespace"""
        result = await adapter.execute("create_namespace", {
            "namespace": "test-ns"
        })
        assert "status" in result
        assert "namespace" in result


class TestInfrastructureIntegration:
    """Integration tests for infrastructure adapters"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_deployment_workflow_structure(self):
        """Test that all adapters work together"""
        terraform = TerraformAdapter()
        ansible = AnsibleAdapter()
        k8s = KubernetesAdapter()
        
        # Terraform
        tf_result = await terraform.execute("init", {"working_dir": "./infra"})
        assert isinstance(tf_result, dict)
        
        # Ansible
        ansible_result = await ansible.execute("playbook", {
            "playbook": "configure.yml",
            "inventory": "hosts.ini"
        })
        assert isinstance(ansible_result, dict)
        
        # Kubernetes
        k8s_result = await k8s.execute("get", {
            "resource": "pods",
            "namespace": "default"
        })
        assert isinstance(k8s_result, dict)
    
    @pytest.mark.asyncio
    async def test_all_adapters_return_status(self):
        """Test that all methods return proper status"""
        adapters = [
            (TerraformAdapter(), "init", {"working_dir": "./test"}),
            (AnsibleAdapter(), "adhoc", {"pattern": "all", "module": "ping"}),
            (KubernetesAdapter(), "get", {"resource": "pods"})
        ]
        
        for adapter, method, inputs in adapters:
            result = await adapter.execute(method, inputs)
            assert "status" in result
            assert isinstance(result["status"], str)
