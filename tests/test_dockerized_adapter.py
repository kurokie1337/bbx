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

import pytest
from unittest.mock import MagicMock, patch
from blackbox.core.base_adapter import DockerizedAdapter

class TestDockerizedAdapter:
    
    @pytest.fixture
    def adapter(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            return DockerizedAdapter(
                adapter_name="TestAdapter",
                docker_image="test/image:latest",
                cli_tool="test-tool"
            )

    @patch("subprocess.run")
    def test_run_command_constructs_docker_cmd(self, mock_run, adapter):
        """Test that run_command constructs the correct docker run command"""
        # Mock successful execution
        mock_run.return_value = MagicMock(returncode=0, stdout='{"status": "ok"}', stderr="")
        
        # Mock image check
        with patch.object(adapter, "_image_exists", return_value=True):
            adapter.run_command("arg1", "arg2")
        
        # Verify call args
        args, kwargs = mock_run.call_args
        cmd = args[0]
        
        # Check basic docker run structure
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        
        # Check volume mount
        assert "-v" in cmd
        assert "-w" in cmd
        
        # Check image and args
        assert "test/image:latest" in cmd
        assert "arg1" in cmd
        assert "arg2" in cmd

    @patch("subprocess.run")
    def test_run_command_with_env_vars(self, mock_run, adapter):
        """Test passing environment variables"""
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        
        with patch.object(adapter, "_image_exists", return_value=True):
            adapter.run_command("test", env={"KEY": "VALUE"})
        
        args, _ = mock_run.call_args
        cmd = args[0]
        
        assert "-e" in cmd
        assert "KEY=VALUE" in cmd

    @patch("subprocess.run")
    def test_run_command_permission_fix(self, mock_run, adapter):
        """Test permission fix logic"""
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        
        # Mock platform.system to return Linux
        with patch("platform.system", return_value="Linux"):
            with patch("os.getuid", return_value=1000):
                with patch("os.getgid", return_value=1000):
                    with patch.object(adapter, "_image_exists", return_value=True):
                        adapter.run_command("test")
        
        args, _ = mock_run.call_args
        cmd = args[0]
        
        # Should have --user 1000:1000
        assert "--user" in cmd
        assert "1000:1000" in cmd

    @patch("subprocess.run")
    def test_image_pull_log(self, mock_run, adapter):
        """Test logging when image needs pulling"""
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        
        # Mock logger
        adapter.logger = MagicMock()
        
        # Mock image check returning False (image missing)
        with patch.object(adapter, "_image_exists", return_value=False):
            adapter.run_command("test")
            
        # Should log info about pulling
        adapter.logger.info.assert_called_with("Pulling Docker image test/image:latest... (this may take a while)")
