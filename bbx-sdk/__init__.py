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
BBX Python SDK - Official Python client for Blackbox Workflow Engine API

This SDK provides a convenient way to interact with the BBX API for workflow
management, execution, and synchronization.

Example:
    >>> from bbx_sdk import BlackboxClient
    >>> client = BlackboxClient("http://localhost:8000")
    >>> client.authenticate("username", "password")
    >>> workflows = client.list_workflows()
"""

__version__ = "1.0.0"
__author__ = "Ilya Makarov"
__license__ = "Apache-2.0"

from .client import BlackboxClient
from .models import (
    WorkflowStatus,
    LicenseType,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowListResponse,
    ExecutionCreate,
    ExecutionResponse,
    AuthLogin,
    AuthResponse,
)
from .sync import WorkflowSyncer
from .config import settings, get_settings

__all__ = [
    "BlackboxClient",
    "WorkflowSyncer",
    "WorkflowStatus",
    "LicenseType",
    "WorkflowCreate",
    "WorkflowUpdate",
    "WorkflowResponse",
    "WorkflowListResponse",
    "ExecutionCreate",
    "ExecutionResponse",
    "AuthLogin",
    "AuthResponse",
    "settings",
    "get_settings",
]
