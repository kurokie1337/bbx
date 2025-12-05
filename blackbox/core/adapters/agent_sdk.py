# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Claude Agent SDK Adapter for BBX Workflows

Enables workflows to invoke Claude Code agents via the official Agent SDK.
Supports subagents, custom tools, and full Claude Code capabilities.

Usage in workflow:
    steps:
      # Simple query
      analyze:
        use: agent.query
        args:
          prompt: "Analyze this code for bugs"

      # Query with options
      generate:
        use: agent.query
        args:
          prompt: "Generate unit tests for ${inputs.file}"
          system_prompt: "You are a testing expert"
          max_turns: 5
          allowed_tools:
            - Read
            - Write
            - Bash

      # Invoke specific subagent
      review:
        use: agent.subagent
        args:
          name: code-reviewer
          prompt: "Review ${steps.generate.output}"

      # List available subagents
      list:
        use: agent.list_subagents
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from blackbox.core.base_adapter import MCPAdapter

logger = logging.getLogger(__name__)

# Check for SDK availability
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("claude-agent-sdk not installed. Install with: pip install claude-agent-sdk")


class AgentSDKAdapter(MCPAdapter):
    """
    Adapter for Claude Agent SDK integration.

    Provides access to Claude Code agents from BBX workflows.
    Supports:
        - Simple queries
        - Custom system prompts
        - Subagent invocation
        - Tool restrictions
        - Context passing
        - RAG enrichment from memory
    """

    def __init__(self):
        super().__init__("agent")
        self._context_tiering = None
        self._rag = None
        self._rag_enabled = True

    def set_context_tiering(self, tiering):
        """Set ContextTiering instance for memory management."""
        self._context_tiering = tiering

    def set_rag(self, rag_instance):
        """Set RAGEnrichment instance for automatic context enrichment."""
        self._rag = rag_instance

    def enable_rag(self):
        """Enable RAG enrichment."""
        self._rag_enabled = True

    def disable_rag(self):
        """Disable RAG enrichment."""
        self._rag_enabled = False

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute Agent SDK adapter method.

        Args:
            method: Method name (query, subagent, list_subagents, etc.)
            inputs: Input parameters

        Returns:
            Method result
        """
        if not SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        self.log_execution(method, inputs)

        method_map = {
            "query": self.query,
            "query_rag": self.query_with_rag,
            "subagent": self.invoke_subagent,
            "list_subagents": self.list_subagents,
            "parallel": self.parallel_query,
            "with_memory": self.query_with_memory,
        }

        handler = method_map.get(method)
        if not handler:
            raise ValueError(
                f"Unknown agent method: {method}. "
                f"Available: {list(method_map.keys())}"
            )

        try:
            result = await handler(**inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            raise

    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: str = "default",
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query Claude via Agent SDK.

        Args:
            prompt: The prompt to send to Claude
            system_prompt: Optional custom system prompt
            max_turns: Maximum agent turns
            allowed_tools: List of allowed tools (Read, Write, Bash, etc.)
            permission_mode: Permission mode (default, acceptEdits, etc.)
            working_dir: Working directory for the agent
            timeout: Query timeout in seconds

        Returns:
            Query result with response and metadata
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions

        # Build options
        options_kwargs = {}

        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt
        if max_turns:
            options_kwargs["max_turns"] = max_turns
        if allowed_tools:
            options_kwargs["allowed_tools"] = allowed_tools
        if permission_mode != "default":
            options_kwargs["permission_mode"] = permission_mode
        if working_dir:
            options_kwargs["cwd"] = working_dir

        options = ClaudeAgentOptions(**options_kwargs) if options_kwargs else None

        # Collect response
        response_parts = []
        tool_calls = []

        try:
            if timeout:
                # With timeout
                async def run_query():
                    async for message in sdk_query(prompt=prompt, options=options):
                        if hasattr(message, 'content'):
                            for block in message.content:
                                if hasattr(block, 'text'):
                                    response_parts.append(block.text)
                                elif hasattr(block, 'name'):  # Tool use
                                    tool_calls.append({
                                        "tool": block.name,
                                        "input": getattr(block, 'input', {})
                                    })

                await asyncio.wait_for(run_query(), timeout=timeout)
            else:
                async for message in sdk_query(prompt=prompt, options=options):
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)
                            elif hasattr(block, 'name'):
                                tool_calls.append({
                                    "tool": block.name,
                                    "input": getattr(block, 'input', {})
                                })

            response = "\n".join(response_parts)

            return {
                "status": "success",
                "response": response,
                "tool_calls": tool_calls,
                "prompt": prompt,
            }

        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"Query timed out after {timeout} seconds",
                "partial_response": "\n".join(response_parts) if response_parts else None,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    async def query_with_rag(
        self,
        prompt: str,
        rag_top_k: int = 5,
        rag_threshold: float = 0.3,
        **query_kwargs,
    ) -> Dict[str, Any]:
        """
        Query with RAG (Retrieval-Augmented Generation).

        Automatically enriches the prompt with relevant context from memory
        before sending to Claude.

        Args:
            prompt: The prompt to send
            rag_top_k: Number of memories to retrieve
            rag_threshold: Minimum relevance threshold
            **query_kwargs: Additional arguments passed to query()

        Returns:
            Query result with RAG metadata
        """
        enriched_prompt = prompt
        rag_metadata = {
            "rag_enabled": False,
            "memories_used": 0,
            "sources": [],
        }

        # Try RAG enrichment if enabled and available
        if self._rag_enabled and self._rag:
            try:
                from blackbox.core.v2.rag_enrichment import RAGConfig

                config = RAGConfig(
                    top_k=rag_top_k,
                    min_relevance=rag_threshold,
                )

                result = await self._rag.enrich(prompt, config=config)

                if result.context_added:
                    enriched_prompt = result.enriched_prompt
                    rag_metadata = {
                        "rag_enabled": True,
                        "memories_found": result.memories_found,
                        "memories_used": result.memories_used,
                        "sources": result.sources,
                        "search_time_ms": result.search_time_ms,
                    }
                    logger.info(
                        f"RAG enriched prompt with {result.memories_used} memories"
                    )

            except Exception as e:
                logger.warning(f"RAG enrichment failed: {e}")

        # Execute query with enriched prompt
        query_result = await self.query(prompt=enriched_prompt, **query_kwargs)

        # Add RAG metadata to result
        query_result["rag"] = rag_metadata
        query_result["original_prompt"] = prompt

        return query_result

    async def invoke_subagent(
        self,
        name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Invoke a specific subagent by name.

        Args:
            name: Subagent name (as defined in .claude/agents/*.md)
            prompt: Task prompt for the subagent
            context: Additional context to provide
            timeout: Execution timeout

        Returns:
            Subagent result
        """
        # Build prompt that explicitly invokes the subagent
        full_prompt = f"Use the {name} subagent to: {prompt}"

        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            full_prompt = f"{full_prompt}\n\nContext:\n{context_str}"

        return await self.query(
            prompt=full_prompt,
            timeout=timeout,
        )

    async def list_subagents(
        self,
        path: str = ".claude/agents",
    ) -> Dict[str, Any]:
        """
        List available subagents.

        Args:
            path: Path to agents directory

        Returns:
            List of subagent definitions
        """
        import os
        from pathlib import Path

        agents_path = Path(path)
        subagents = []

        if agents_path.exists():
            for file in agents_path.glob("*.md"):
                content = file.read_text(encoding="utf-8")

                # Parse frontmatter
                name = file.stem
                description = ""
                tools = []

                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        import yaml
                        try:
                            frontmatter = yaml.safe_load(parts[1])
                            name = frontmatter.get("name", name)
                            description = frontmatter.get("description", "")
                            tools = frontmatter.get("tools", [])
                            if isinstance(tools, str):
                                tools = [t.strip() for t in tools.split(",")]
                        except:
                            pass

                subagents.append({
                    "name": name,
                    "file": str(file),
                    "description": description,
                    "tools": tools,
                })

        return {
            "status": "success",
            "subagents": subagents,
            "count": len(subagents),
            "path": str(agents_path),
        }

    async def parallel_query(
        self,
        queries: List[Dict[str, Any]],
        max_parallel: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute multiple queries in parallel.

        Args:
            queries: List of query configurations
            max_parallel: Maximum parallel executions

        Returns:
            Results from all queries
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_query(idx: int, query_config: Dict) -> Dict:
            async with semaphore:
                result = await self.query(**query_config)
                return {"index": idx, "result": result}

        tasks = [
            run_query(i, q) for i, q in enumerate(queries)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = []
        failed = []

        for r in results:
            if isinstance(r, Exception):
                failed.append({"error": str(r)})
            elif r.get("result", {}).get("status") == "success":
                successful.append(r)
            else:
                failed.append(r)

        return {
            "status": "completed",
            "total": len(queries),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
        }

    async def query_with_memory(
        self,
        prompt: str,
        memory_key: str,
        include_history: bool = True,
        max_history: int = 10,
        **query_kwargs,
    ) -> Dict[str, Any]:
        """
        Query with memory from ContextTiering.

        Args:
            prompt: The prompt
            memory_key: Key for storing/retrieving memory
            include_history: Whether to include conversation history
            max_history: Maximum history entries to include
            **query_kwargs: Additional query arguments

        Returns:
            Query result
        """
        # Get history from ContextTiering if available
        history = []
        if self._context_tiering and include_history:
            try:
                stored = await self._context_tiering.get(memory_key)
                if stored and isinstance(stored, list):
                    history = stored[-max_history:]
            except:
                pass

        # Build prompt with history
        if history:
            history_str = "\n".join(
                f"[{h.get('role', 'user')}]: {h.get('content', '')}"
                for h in history
            )
            full_prompt = f"Previous conversation:\n{history_str}\n\nNew request: {prompt}"
        else:
            full_prompt = prompt

        # Execute query
        result = await self.query(prompt=full_prompt, **query_kwargs)

        # Store in memory
        if self._context_tiering and result.get("status") == "success":
            try:
                new_entry = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result.get("response", "")},
                ]
                updated_history = history + new_entry
                await self._context_tiering.set(memory_key, updated_history[-max_history * 2:])
            except:
                pass

        return result


# =============================================================================
# A2A Wrapper for Agent SDK
# =============================================================================

class AgentA2AWrapper:
    """
    Wraps Agent SDK as A2A-compatible service.

    Allows Claude Code agents to be discovered and called via A2A protocol.
    """

    def __init__(
        self,
        name: str = "Claude Code Agent",
        port: int = 9000,
        skills: Optional[List[Dict]] = None,
    ):
        self.name = name
        self.port = port
        self.adapter = AgentSDKAdapter()
        self.skills = skills or [
            {
                "id": "query",
                "name": "Query Claude",
                "description": "Send a query to Claude Code",
            },
            {
                "id": "subagent",
                "name": "Invoke Subagent",
                "description": "Invoke a specialized subagent",
            },
        ]

    def get_agent_card(self) -> Dict[str, Any]:
        """Return A2A Agent Card."""
        return {
            "name": self.name,
            "description": "Claude Code agent via Agent SDK",
            "url": f"http://localhost:{self.port}",
            "version": "1.0.0",
            "protocolVersion": "0.3",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
            },
            "skills": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "description": s["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                        },
                        "required": ["prompt"],
                    }
                }
                for s in self.skills
            ],
            "authentication": {"schemes": ["none"]},
            "tags": ["claude-code", "agent-sdk"],
        }

    async def handle_task(self, skill_id: str, input_data: Dict) -> Dict:
        """Handle A2A task by delegating to Agent SDK."""
        if skill_id == "query":
            return await self.adapter.query(**input_data)
        elif skill_id == "subagent":
            return await self.adapter.invoke_subagent(**input_data)
        else:
            raise ValueError(f"Unknown skill: {skill_id}")

    def create_fastapi_app(self):
        """Create FastAPI app for A2A server."""
        from fastapi import FastAPI, BackgroundTasks
        from fastapi.responses import JSONResponse
        import uuid
        from datetime import datetime, timezone

        app = FastAPI(title=self.name)
        tasks = {}

        @app.get("/.well-known/agent-card.json")
        async def agent_card():
            return self.get_agent_card()

        @app.post("/a2a/tasks")
        async def create_task(task_input: dict, background_tasks: BackgroundTasks):
            task_id = str(uuid.uuid4())
            skill_id = task_input.get("skillId")
            input_data = task_input.get("input", {})

            task = {
                "id": task_id,
                "skillId": skill_id,
                "input": input_data,
                "status": "pending",
                "output": None,
                "error": None,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
            tasks[task_id] = task

            async def execute():
                task["status"] = "in_progress"
                try:
                    result = await self.handle_task(skill_id, input_data)
                    task["status"] = "completed"
                    task["output"] = result
                except Exception as e:
                    task["status"] = "failed"
                    task["error"] = str(e)
                task["completedAt"] = datetime.now(timezone.utc).isoformat()

            background_tasks.add_task(execute)
            return task

        @app.get("/a2a/tasks/{task_id}")
        async def get_task(task_id: str):
            task = tasks.get(task_id)
            if not task:
                return JSONResponse(status_code=404, content={"error": "Not found"})
            return task

        @app.get("/health")
        async def health():
            return {"status": "healthy", "agent": self.name}

        return app
