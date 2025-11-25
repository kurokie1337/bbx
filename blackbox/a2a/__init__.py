# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX A2A Protocol Implementation

Google Agent2Agent (A2A) protocol support for BBX.
Enables BBX to participate in multi-agent ecosystems.

Components:
- AgentCard: Capability advertisement
- A2AServer: HTTP server for receiving tasks
- A2AClient: Client for calling other A2A agents
- A2AAdapter: Workflow adapter for a2a.call
"""

from .models import (
    AgentCard,
    AgentSkill,
    A2ATask,
    A2ATaskStatus,
    A2AMessage,
    A2AArtifact,
)
from .agent_card import AgentCardGenerator
from .client import A2AClient
from .server import create_a2a_app

__all__ = [
    # Models
    "AgentCard",
    "AgentSkill",
    "A2ATask",
    "A2ATaskStatus",
    "A2AMessage",
    "A2AArtifact",
    # Components
    "AgentCardGenerator",
    "A2AClient",
    "create_a2a_app",
]
