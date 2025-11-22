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
from unittest.mock import Mock, patch
from uuid import uuid4
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "bbx-sdk"))

from bbx_sdk import BlackboxClient, WorkflowCreate

@pytest.fixture
def mock_client():
    with patch('httpx.Client') as mock_cls:
        # Mock the instance returned by the constructor
        mock_instance = Mock()
        mock_instance.headers = {} # Use a real dict for headers
        mock_cls.return_value = mock_instance
        yield mock_instance

def test_authenticate_success(mock_client):
    # Setup
    client = BlackboxClient()
    # client.client is now mock_client (the instance)
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "access_token": "fake_token",
        "token_type": "bearer",
        "expires_in": 3600
    }
    mock_response.status_code = 200
    mock_client.post.return_value = mock_response

    # Execute
    auth = client.authenticate("user", "pass")

    # Verify
    assert auth.access_token == "fake_token"
    assert client.token == "fake_token"
    assert client.client.headers["Authorization"] == "Bearer fake_token"

def test_create_workflow(mock_client):
    # Setup
    client = BlackboxClient()
    client.token = "fake_token"
    
    wf_id = str(uuid4())
    user_id = str(uuid4())
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "success",
        "data": {
            "id": wf_id,
            "user_id": user_id,
            "name": "Test Flow",
            "description": "Test Description",
            "bbx_yaml": "...",
            "status": "draft",
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    mock_response.status_code = 201
    mock_client.post.return_value = mock_response

    # Execute
    wf = WorkflowCreate(name="Test Flow", bbx_yaml="...")
    result = client.create_workflow(wf)

    # Verify
    assert result.name == "Test Flow"
    assert str(result.id) == wf_id
