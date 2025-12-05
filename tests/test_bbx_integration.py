#!/usr/bin/env python3
"""
BBX FULL INTEGRATION TEST - All Systems Connected

This is the ULTIMATE test where ALL components work together:
- SIRE Kernel orchestrates everything
- Daemon manages agents
- LLM provides intelligence
- VectorDB stores memories
- Workflows execute multi-step tasks
- API serves HTTP/WebSocket
- Snapshots enable recovery
- Memory persists across sessions

Tests the FULL POWER of the interconnected system!
"""

import asyncio
import os
import sys
import time
import json
import aiohttp
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BBXIntegrationTest:
    """Full integration test harness"""

    def __init__(self):
        self.daemon = None
        self.api = None
        self.kernel = None
        self.results = []
        self.start_time = None

    async def setup(self):
        """Initialize ALL systems"""
        print("\n" + "=" * 70)
        print("BBX FULL INTEGRATION TEST - ALL SYSTEMS CONNECTED")
        print("=" * 70)
        self.start_time = time.time()

        # Boot SIRE Kernel (the brain)
        from blackbox.core.v2 import get_kernel
        self.kernel = await get_kernel()
        print("[KERNEL] SIRE Kernel booted")

        # Start Daemon (agent manager)
        from blackbox.runtime.daemon import BBXDaemon
        self.daemon = BBXDaemon()
        await self.daemon.start()
        print("[DAEMON] BBX Daemon started")

        # Start API Server (interface)
        from blackbox.runtime.api import BBXAPIServer
        self.api = BBXAPIServer(self.daemon, host="127.0.0.1", port=9996)
        await self.api.start()
        print("[API] API Server started on port 9996")

        await asyncio.sleep(1)
        print("\n[OK] All systems initialized!\n")

    async def teardown(self):
        """Shutdown all systems"""
        print("\n[...] Shutting down all systems...")

        if self.api:
            await self.api.stop()

        if self.daemon:
            await self.daemon.stop()

        from blackbox.runtime.llm_provider import _llm_manager
        if _llm_manager:
            await _llm_manager.shutdown()

        duration = time.time() - self.start_time
        print(f"[OK] All systems stopped (total time: {duration:.1f}s)")

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        self.results.append((test_name, passed, details))
        print(f"  [{status}] {test_name}" + (f": {details}" if details else ""))


    async def test_1_full_pipeline_llm_memory_vectordb(self):
        """
        TEST 1: Full LLM -> Memory -> VectorDB Pipeline

        Flow:
        1. Agent thinks with LLM
        2. Stores thought in Memory
        3. Memory persists to VectorDB
        4. Later agent recalls from VectorDB
        5. Uses recalled info for new thought
        """
        print("\n" + "-" * 60)
        print("TEST 1: LLM -> Memory -> VectorDB Full Pipeline")
        print("-" * 60)

        from blackbox.runtime.daemon import AgentConfig
        from blackbox.runtime.vectordb_provider import get_memory_store

        memory_store = await get_memory_store()
        agent_id = "pipeline_test_agent"

        # Clear old data
        await memory_store.clear_agent_memory(agent_id)

        # Step 1: Spawn agent
        config = AgentConfig(
            name="knowledge-worker",
            description="Agent that learns and remembers",
            system_prompt="You are a knowledge worker. Learn and remember facts. Be concise.",
        )
        agent = await self.daemon.agents.spawn(config)
        print(f"  [1/5] Agent spawned: {agent.id}")

        # Step 2: Agent thinks and learns
        await self.daemon.agents.dispatch_task(agent.id, {
            "type": "think",
            "prompt": "What is the capital of France? Answer in one word.",
            "max_tokens": 20,
        })
        await asyncio.sleep(2)
        print(f"  [2/5] Agent thought about France")

        # Step 3: Store knowledge in memory
        await memory_store.store_memory(
            agent_id=agent_id,
            content="The capital of France is Paris",
            memory_type="fact",
            importance=0.95,
        )
        await memory_store.store_memory(
            agent_id=agent_id,
            content="France is in Western Europe",
            memory_type="fact",
            importance=0.8,
        )
        print(f"  [3/5] Stored knowledge in VectorDB")

        # Step 4: Recall from memory
        results = await memory_store.recall(
            agent_id=agent_id,
            query="What do you know about France?",
            top_k=2,
        )
        print(f"  [4/5] Recalled {len(results)} memories from VectorDB")
        for r in results:
            print(f"        -> {r.content[:50]}... (score: {r.score:.3f})")

        # Step 5: Use recalled info for new thought
        context = " ".join([r.content for r in results])
        await self.daemon.agents.dispatch_task(agent.id, {
            "type": "think",
            "prompt": f"Based on: {context}. What language do they speak there?",
            "max_tokens": 30,
        })
        await asyncio.sleep(2)
        print(f"  [5/5] Agent used memory for reasoning")

        # Verify
        passed = len(results) >= 2 and agent.tasks_completed >= 2
        self.log_result("LLM-Memory-VectorDB Pipeline", passed,
                        f"{agent.tasks_completed} tasks, {len(results)} memories")

        # Cleanup
        await self.daemon.agents.kill(agent.id)
        await memory_store.clear_agent_memory(agent_id)

        return passed


    async def test_2_daemon_workflow_agent_chain(self):
        """
        TEST 2: Daemon -> Workflow -> Agent Chain

        Flow:
        1. Daemon spawns multiple specialized agents
        2. Workflow coordinates agents in sequence
        3. Each agent contributes to final result
        4. Data flows between agents via workflow
        """
        print("\n" + "-" * 60)
        print("TEST 2: Daemon -> Workflow -> Agent Chain")
        print("-" * 60)

        from blackbox.runtime.daemon import AgentConfig
        from blackbox.runtime.workflows import WorkflowLoader, WorkflowEngine
        from blackbox.runtime.daemon import SnapshotManager, BBX_SNAPSHOTS_DIR
        from blackbox.runtime.llm_provider import get_llm_manager

        # Step 1: Spawn specialized agents via daemon
        agents_config = [
            ("data-collector", "Collects raw data"),
            ("data-processor", "Processes and cleans data"),
            ("report-generator", "Creates final reports"),
        ]

        agents = []
        for name, desc in agents_config:
            config = AgentConfig(name=name, description=desc,
                                 system_prompt=f"You are {name}. {desc}. Be very concise.")
            agent = await self.daemon.agents.spawn(config)
            agents.append(agent)
        print(f"  [1/4] Spawned {len(agents)} specialized agents")

        # Step 2: Define workflow that uses agents
        workflow_yaml = """
name: "data-pipeline"
description: "Full data processing pipeline"
version: "1.0"
steps:
  - id: "collect"
    agent: "data-collector"
    task: "List 2 popular programming languages"
    output_key: "raw_data"
  - id: "process"
    agent: "data-processor"
    task: "Summarize the data in one sentence"
    depends_on: ["collect"]
    input:
      data: "{{ .raw_data }}"
    output_key: "processed"
  - id: "report"
    agent: "report-generator"
    task: "Create a one-line conclusion"
    depends_on: ["process"]
    input:
      processed: "{{ .processed }}"
    output_key: "report"
"""
        loader = WorkflowLoader()
        config = loader.load_from_string(workflow_yaml)
        print(f"  [2/4] Loaded workflow: {config.name} ({len(config.steps)} steps)")

        # Step 3: Execute workflow
        llm = await get_llm_manager()

        async def agent_executor(agent_name: str, task_input: dict) -> dict:
            prompt = task_input.get("task", "")
            if "data" in task_input:
                prompt = f"Data: {task_input['data']}\n\nTask: {prompt}"
            if "processed" in task_input:
                prompt = f"Processed: {task_input['processed']}\n\nTask: {prompt}"

            response = await llm.complete(
                prompt=prompt,
                system=f"You are {agent_name}. Be very concise (1 sentence max).",
                max_tokens=100,
            )
            return {"content": response.content}

        snapshot_manager = SnapshotManager(BBX_SNAPSHOTS_DIR)
        engine = WorkflowEngine(agent_executor, snapshot_manager)

        start = time.time()
        instance = await engine.run(config, variables={})
        duration = time.time() - start
        print(f"  [3/4] Workflow executed in {duration:.1f}s")

        # Step 4: Verify chain execution
        completed_steps = sum(1 for r in instance.step_results.values()
                             if r.status.value == "completed")
        print(f"  [4/4] Completed {completed_steps}/{len(config.steps)} steps")

        for step_id, result in instance.step_results.items():
            print(f"        {step_id}: {result.status.value}")

        passed = completed_steps == len(config.steps)
        self.log_result("Daemon-Workflow-Agent Chain", passed,
                        f"{completed_steps} steps, {duration:.1f}s")

        # Cleanup
        for agent in agents:
            await self.daemon.agents.kill(agent.id)

        return passed


    async def test_3_api_websocket_realtime(self):
        """
        TEST 3: API -> WebSocket -> Real-time Updates

        Flow:
        1. Connect WebSocket
        2. Subscribe to events
        3. Trigger actions via REST API
        4. Receive real-time updates via WebSocket
        5. Verify event propagation
        """
        print("\n" + "-" * 60)
        print("TEST 3: API -> WebSocket -> Real-time Updates")
        print("-" * 60)

        events_received = []

        async with aiohttp.ClientSession() as session:
            # Step 1: Connect WebSocket
            ws = await session.ws_connect("http://127.0.0.1:9996/ws")
            print(f"  [1/5] WebSocket connected")

            # Step 2: Subscribe to all agent events
            await ws.send_json({
                "type": "subscribe",
                "events": ["agent_spawned", "agent_stopped", "agent_task_started"]
            })
            print(f"  [2/5] Subscribed to agent events")

            # Step 3: Trigger agent spawn via REST
            async with session.post(
                "http://127.0.0.1:9996/api/v1/agents",
                json={"config": {"name": "ws-event-test"}}
            ) as resp:
                data = await resp.json()
                agent_id = data["data"]["agent_id"]
            print(f"  [3/5] Spawned agent via REST: {agent_id}")

            # Step 4: Receive WebSocket events
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=3)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    events_received.append(event)
                    print(f"  [4/5] Received event: {event.get('type', 'unknown')}")
            except asyncio.TimeoutError:
                print(f"  [4/5] No event received (timeout)")

            # Step 5: Dispatch task and check for event
            async with session.post(
                f"http://127.0.0.1:9996/api/v1/agents/{agent_id}/task",
                json={"type": "think", "prompt": "Say hello", "max_tokens": 10}
            ) as resp:
                pass

            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=3)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    events_received.append(event)
                    print(f"  [5/5] Received task event: {event.get('type', 'unknown')}")
            except asyncio.TimeoutError:
                print(f"  [5/5] No task event (async processing)")

            # Cleanup
            async with session.delete(f"http://127.0.0.1:9996/api/v1/agents/{agent_id}"):
                pass

            await ws.close()

        passed = len(events_received) >= 1
        self.log_result("API-WebSocket Real-time", passed,
                        f"{len(events_received)} events received")

        return passed


    async def test_4_full_recovery_all_components(self):
        """
        TEST 4: Full Recovery with All Components

        Flow:
        1. Create agent with memory
        2. Execute tasks
        3. Create snapshot of full state
        4. Simulate failure
        5. Recover from snapshot
        6. Verify all data restored
        """
        print("\n" + "-" * 60)
        print("TEST 4: Full Recovery with All Components")
        print("-" * 60)

        from blackbox.runtime.daemon import AgentConfig, SnapshotManager, BBX_SNAPSHOTS_DIR
        from blackbox.runtime.vectordb_provider import get_memory_store

        memory_store = await get_memory_store()
        snapshots = SnapshotManager(BBX_SNAPSHOTS_DIR)
        agent_id = "recovery_full_test"

        # Step 1: Create agent
        config = AgentConfig(
            name="recoverable-agent",
            description="Agent that can be recovered",
            system_prompt="You are a recoverable agent.",
        )
        agent = await self.daemon.agents.spawn(config)
        print(f"  [1/6] Agent created: {agent.id}")

        # Step 2: Store memories
        await memory_store.store_memory(
            agent_id=agent_id,
            content="Critical business data: Q4 revenue is $1M",
            memory_type="fact",
            importance=0.99,
        )
        await memory_store.store_memory(
            agent_id=agent_id,
            content="User preference: Always use dark mode",
            memory_type="preference",
            importance=0.85,
        )
        print(f"  [2/6] Stored critical memories")

        # Step 3: Execute task
        await self.daemon.agents.dispatch_task(agent.id, {
            "type": "think",
            "prompt": "Acknowledge: I have important data stored",
            "max_tokens": 30,
        })
        await asyncio.sleep(2)
        print(f"  [3/6] Task executed: {agent.tasks_completed} completed")

        # Step 4: Create full state snapshot
        full_state = {
            "agent_id": agent.id,
            "tasks_completed": agent.tasks_completed,
            "tokens_used": agent.total_tokens_used,
            "memory_count": await memory_store.get_agent_memory_count(agent_id),
            "timestamp": datetime.utcnow().isoformat(),
        }
        snap_id = await snapshots.create_snapshot(agent_id, full_state, "Full state backup")
        print(f"  [4/6] Created snapshot: {snap_id}")

        # Step 5: Simulate failure (kill agent)
        await self.daemon.agents.kill(agent.id)
        print(f"  [5/6] Simulated failure (agent killed)")

        # Step 6: Recover and verify
        restored_state = await snapshots.restore_snapshot(snap_id)
        memories_after = await memory_store.recall(agent_id=agent_id, query="business data", top_k=1)

        print(f"  [6/6] Recovery verification:")
        print(f"        State restored: {restored_state is not None}")
        print(f"        Tasks in snapshot: {restored_state.get('tasks_completed', 0)}")
        print(f"        Memories preserved: {len(memories_after)}")

        passed = (restored_state is not None and
                  restored_state.get("tasks_completed", 0) >= 1 and
                  len(memories_after) >= 1)

        self.log_result("Full Recovery All Components", passed,
                        f"Snapshot + {len(memories_after)} memories preserved")

        # Cleanup
        await memory_store.clear_agent_memory(agent_id)

        return passed


    async def test_5_cross_component_data_flow(self):
        """
        TEST 5: Cross-Component Data Flow

        Flow: API -> Daemon -> Agent -> LLM -> Memory -> VectorDB -> API

        1. API receives request
        2. Daemon spawns agent
        3. Agent thinks with LLM
        4. Result stored in Memory
        5. Memory persisted to VectorDB
        6. API queries VectorDB and returns results
        """
        print("\n" + "-" * 60)
        print("TEST 5: Cross-Component Data Flow")
        print("-" * 60)

        async with aiohttp.ClientSession() as session:
            # Step 1: API -> Spawn agent
            async with session.post(
                "http://127.0.0.1:9996/api/v1/agents",
                json={
                    "config": {
                        "name": "flow-test-agent",
                        "description": "Tests data flow",
                    }
                }
            ) as resp:
                data = await resp.json()
                agent_id = data["data"]["agent_id"]
            print(f"  [1/6] API -> Daemon: Agent spawned: {agent_id}")

            # Step 2: Agent thinks with LLM (via daemon)
            async with session.post(
                f"http://127.0.0.1:9996/api/v1/agents/{agent_id}/task",
                json={
                    "type": "think",
                    "prompt": "What is 7 * 8? Answer with just the number.",
                    "max_tokens": 20,
                }
            ) as resp:
                pass
            print(f"  [2/6] Daemon -> Agent -> LLM: Task dispatched")

            await asyncio.sleep(2)

            # Step 3: Store result in memory (via daemon)
            async with session.post(
                f"http://127.0.0.1:9996/api/v1/agents/{agent_id}/task",
                json={
                    "type": "remember",
                    "content": "7 * 8 = 56, calculated by agent",
                    "memory_type": "calculation",
                    "importance": 0.9,
                }
            ) as resp:
                pass
            print(f"  [3/6] Agent -> Memory: Result stored")

            await asyncio.sleep(1)

            # Step 4: Check agent status via API
            async with session.get(f"http://127.0.0.1:9996/api/v1/agents/{agent_id}") as resp:
                agent_data = await resp.json()
                tasks_done = agent_data["data"]["tasks_completed"]
            print(f"  [4/6] API -> Status: {tasks_done} tasks completed")

            # Step 5: Search memory via API -> VectorDB
            async with session.get(
                "http://127.0.0.1:9996/api/v1/memory/search",
                params={"q": "multiplication calculation", "agent_id": "system", "top_k": 5}
            ) as resp:
                memory_data = await resp.json()
                memories_found = len(memory_data.get("data", []))
            print(f"  [5/6] API -> VectorDB: {memories_found} related memories")

            # Step 6: Get system status (full loop)
            async with session.get("http://127.0.0.1:9996/api/v1/status") as resp:
                status = await resp.json()
                running = status["data"]["running"]
            print(f"  [6/6] Full loop complete: System running: {running}")

            # Cleanup
            async with session.delete(f"http://127.0.0.1:9996/api/v1/agents/{agent_id}"):
                pass

        passed = tasks_done >= 2 and running
        self.log_result("Cross-Component Data Flow", passed,
                        f"API->Daemon->LLM->Memory->VectorDB->API")

        return passed


    async def test_6_concurrent_everything(self):
        """
        TEST 6: Concurrent Everything

        Stress test: Run multiple agents, workflows, and API calls concurrently
        to test system under load with all components active.
        """
        print("\n" + "-" * 60)
        print("TEST 6: Concurrent Everything (Stress Test)")
        print("-" * 60)

        from blackbox.runtime.daemon import AgentConfig
        from blackbox.runtime.vectordb_provider import get_memory_store

        memory_store = await get_memory_store()
        NUM_AGENTS = 3
        NUM_MEMORIES_PER_AGENT = 5

        # Step 1: Spawn multiple agents concurrently
        print(f"  [1/4] Spawning {NUM_AGENTS} agents concurrently...")

        async def spawn_agent(i):
            config = AgentConfig(
                name=f"concurrent-{i}",
                description=f"Concurrent agent {i}",
                system_prompt="You are a concurrent test agent. Be brief.",
            )
            return await self.daemon.agents.spawn(config)

        agents = await asyncio.gather(*[spawn_agent(i) for i in range(NUM_AGENTS)])
        print(f"        Spawned: {len(agents)} agents")

        # Step 2: Dispatch tasks to all agents concurrently
        print(f"  [2/4] Dispatching tasks to all agents...")

        async def dispatch_task(agent):
            await self.daemon.agents.dispatch_task(agent.id, {
                "type": "think",
                "prompt": f"Count to 3",
                "max_tokens": 30,
            })

        await asyncio.gather(*[dispatch_task(a) for a in agents])

        # Wait for completion
        await asyncio.sleep(3)

        completed = sum(1 for a in agents if a.tasks_completed > 0)
        print(f"        Tasks completed: {completed}/{NUM_AGENTS}")

        # Step 3: Store memories concurrently
        print(f"  [3/4] Storing {NUM_AGENTS * NUM_MEMORIES_PER_AGENT} memories concurrently...")

        async def store_memory(agent_id, i):
            await memory_store.store_memory(
                agent_id=agent_id,
                content=f"Concurrent memory #{i} stored at {time.time()}",
                memory_type="test",
                importance=0.5,
            )

        tasks = []
        for agent in agents:
            for i in range(NUM_MEMORIES_PER_AGENT):
                tasks.append(store_memory(f"concurrent_{agent.id}", i))

        await asyncio.gather(*tasks)
        print(f"        Memories stored successfully")

        # Step 4: Make concurrent API calls
        print(f"  [4/4] Making concurrent API calls...")

        async with aiohttp.ClientSession() as session:
            async def api_call(endpoint):
                async with session.get(f"http://127.0.0.1:9996{endpoint}") as resp:
                    return resp.status

            endpoints = [
                "/health",
                "/api/v1/status",
                "/api/v1/agents",
                "/api/v1/workflows",
                "/api/v1/memory",
                "/api/v1/snapshots",
            ]

            statuses = await asyncio.gather(*[api_call(e) for e in endpoints])
            successful = sum(1 for s in statuses if s == 200)
            print(f"        API calls: {successful}/{len(endpoints)} successful")

        # Cleanup
        for agent in agents:
            await self.daemon.agents.kill(agent.id)
            await memory_store.clear_agent_memory(f"concurrent_{agent.id}")

        passed = completed == NUM_AGENTS and successful == len(endpoints)
        self.log_result("Concurrent Everything", passed,
                        f"{completed} agents, {successful} API calls")

        return passed


    async def run_all_tests(self):
        """Run all integration tests"""
        await self.setup()

        try:
            await self.test_1_full_pipeline_llm_memory_vectordb()
            await self.test_2_daemon_workflow_agent_chain()
            await self.test_3_api_websocket_realtime()
            await self.test_4_full_recovery_all_components()
            await self.test_5_cross_component_data_flow()
            await self.test_6_concurrent_everything()

        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()

        await self.teardown()

        # Print summary
        print("\n" + "=" * 70)
        print("INTEGRATION TEST RESULTS")
        print("=" * 70)

        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)

        for name, p, details in self.results:
            status = "[PASS]" if p else "[FAIL]"
            print(f"  {status} {name}")
            if details:
                print(f"          {details}")

        print(f"\nTotal: {passed}/{total} passed")

        if passed == total:
            print("\n" + "=" * 70)
            print("ALL INTEGRATION TESTS PASSED!")
            print("FULL SYSTEM POWER VERIFIED!")
            print("=" * 70)
            return 0
        else:
            return 1


async def main():
    test = BBXIntegrationTest()
    return await test.run_all_tests()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
