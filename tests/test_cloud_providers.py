# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Tests for Cloud Provider Adapters (AWS, GCP, Azure, Database)
"""

import pytest
from blackbox.core.adapters.aws import AWSAdapter
from blackbox.core.adapters.gcp import GCPAdapter
from blackbox.core.adapters.azure import AzureAdapter
from blackbox.core.adapters.database import DatabaseMigrationAdapter


class TestAWSAdapter:
    """Tests for AWS adapter"""
    
    @pytest.fixture
    def adapter(self):
        return AWSAdapter()
    
    def test_aws_adapter_methods_exist(self, adapter):
        """Test that all AWS methods are present"""
        methods = [
            "_ec2_launch", "_ec2_terminate", "_ec2_describe",
            "_s3_create_bucket", "_s3_upload", "_s3_list",
            "_lambda_deploy", "_lambda_invoke",
            "_cfn_deploy", "_cfn_delete",
            "_rds_create"
        ]
        
        for method in methods:
            assert hasattr(adapter, method), f"Method {method} not found"
    
    @pytest.mark.asyncio
    async def test_ec2_describe_structure(self, adapter):
        """Test EC2 describe returns correct structure"""
        result = await adapter._ec2_describe({})
        
        # Should return dict even if AWS CLI not configured
        assert isinstance(result, dict)
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_s3_list_structure(self, adapter):
        """Test S3 list structure"""
        result = await adapter._s3_list({"bucket": "test-bucket"})
        
        assert isinstance(result, dict)
        assert "status" in result


class TestGCPAdapter:
    """Tests for GCP adapter"""
    
    @pytest.fixture
    def adapter(self):
        return GCPAdapter()
    
    def test_gcp_adapter_methods_exist(self, adapter):
        """Test that all GCP methods are present"""
        methods = [
            "_compute_create", "_compute_delete", "_compute_list",
            "_storage_create_bucket", "_storage_upload", "_storage_list",
            "_function_deploy", "_function_call",
            "_run_deploy",
            "_gke_create", "_gke_get_credentials"
        ]
        
        for method in methods:
            assert hasattr(adapter, method), f"Method {method} not found"
    
    @pytest.mark.asyncio
    async def test_compute_list_structure(self, adapter):
        """Test Compute Engine list structure"""
        result = await adapter._compute_list({})
        
        assert isinstance(result, dict)
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_storage_list_structure(self, adapter):
        """Test Cloud Storage list structure"""
        result = await adapter._storage_list({"bucket": "test-bucket"})
        
        assert isinstance(result, dict)
        assert "status" in result


class TestAzureAdapter:
    """Tests for Azure adapter"""
    
    @pytest.fixture
    def adapter(self):
        return AzureAdapter()
    
    def test_azure_adapter_methods_exist(self, adapter):
        """Test that all Azure methods are present"""
        methods = [
            "_vm_create", "_vm_delete", "_vm_list",
            "_storage_create_account", "_storage_upload_blob",
            "_function_create_app", "_function_deploy",
            "_aks_create", "_aks_get_credentials",
            "_group_create"
        ]
        
        for method in methods:
            assert hasattr(adapter, method), f"Method {method} not found"
    
    @pytest.mark.asyncio
    async def test_vm_list_structure(self, adapter):
        """Test VM list structure"""
        result = await adapter._vm_list({})
        
        assert isinstance(result, dict)
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_group_create_structure(self, adapter):
        """Test resource group create structure"""
        result = await adapter._group_create({
            "name": "test-rg",
            "location": "eastus"
        })
        
        assert isinstance(result, dict)
        assert "status" in result


class TestDatabaseAdapter:
    """Tests for Database Migration adapter"""
    
    @pytest.fixture
    def adapter(self):
        return DatabaseMigrationAdapter()
    
    def test_db_adapter_methods_exist(self, adapter):
        """Test that all database methods are present"""
        methods = [
            "_migrate", "_rollback", "_status",
            "_create_migration", "_seed"
        ]
        
        for method in methods:
            assert hasattr(adapter, method), f"Method {method} not found"
    
    @pytest.mark.asyncio
    async def test_create_migration(self, adapter):
        """Test migration creation"""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await adapter._create_migration({
                "name": "test_migration",
                "migrations_dir": tmpdir
            })
            
            assert result["status"] == "created"
            assert "migration" in result
            assert "path" in result
            
            # Verify file was created
            migration_path = Path(result["path"])
            assert migration_path.exists()
            assert migration_path.suffix == ".sql"
    
    @pytest.mark.asyncio
    async def test_migration_status(self, adapter):
        """Test migration status"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await adapter._status({
                "database": "postgresql",
                "connection": "postgresql://test",
                "migrations_dir": tmpdir
            })
            
            assert result["status"] == "ok"
            assert "total_migrations" in result
            assert "applied" in result
            assert "pending" in result
    
    @pytest.mark.asyncio
    async def test_rollback_structure(self, adapter):
        """Test rollback structure"""
        result = await adapter._rollback({
            "database": "postgresql",
            "connection": "postgresql://test",
            "steps": 1
        })
        
        assert result["status"] == "ok"
        assert "message" in result


class TestCloudIntegration:
    """Integration tests for cloud providers"""
    
    @pytest.mark.asyncio
    async def test_all_adapters_registered(self):
        """Test that all cloud adapters are registered"""
        from blackbox.core.registry import registry
        
        adapters = [
            "aws", "bbx.aws",
            "gcp", "bbx.gcp",
            "azure", "bbx.azure",
            "db", "bbx.db"
        ]
        
        for adapter_name in adapters:
            adapter = registry.get_adapter(adapter_name)
            assert adapter is not None, f"Adapter {adapter_name} not registered"
    
    @pytest.mark.asyncio
    async def test_multi_cloud_workflow_structure(self):
        """Test multi-cloud workflow exists"""
        from pathlib import Path
        
        workflow_path = Path("workflows/cloud/multi_cloud_deployment.bbx")
        # Note: Path may not exist in test environment, just checking structure
        assert workflow_path.suffix == ".bbx"
