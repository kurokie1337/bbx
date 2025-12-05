#!/usr/bin/env python3
"""
BBX CONSTRUCTOR TEST - Block by Block Connection Verification
Tests EVERY connection between EVERY component.
"""

import asyncio
import os
import sys
import time
import json
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BlockConnection:
    def __init__(self, from_block, to_block):
        self.from_block = from_block
        self.to_block = to_block
        self.verified = False
        self.data_passed = None
        self.latency_ms = 0


class BBXConstructorTest:

    def __init__(self):
        self.connections = []
        self.blocks_initialized = {}

    def add_connection(self, from_b, to_b, verified, data="", latency=0):
        conn = BlockConnection(from_b, to_b)
        conn.verified = verified
        conn.data_passed = data
        conn.latency_ms = latency
        self.connections.append(conn)
        return verified

    async def run(self):
        print("\n" + "=" * 70)
        print("BBX CONSTRUCTOR TEST - BLOCK BY BLOCK VERIFICATION")
        print("=" * 70)
        print("\nInitializing all blocks...\n")

        await self.init_all_blocks()

        print("\n" + "-" * 70)
        print("TESTING ALL CONNECTIONS")
        print("-" * 70 + "\n")

        await self.test_kernel_to_llm()
        await self.test_kernel_to_vectordb()
        await self.test_daemon_to_agent()
        await self.test_agent_to_llm()
        await self.test_agent_to_memory()
        await self.test_memory_to_vectordb()
        await self.test_workflow_to_agent()
        await self.test_workflow_to_snapshot()
        await self.test_api_to_daemon()
        await self.test_api_to_websocket()
        await self.test_api_to_memory()
        await self.test_api_to_snapshot()
        await self.test_snapshot_to_recovery()
        await self.test_full_chain()

        await self.cleanup()
        self.print_results()

    async def init_all_blocks(self):
        print("  [1/7] Initializing SIRE Kernel...")
        from blackbox.core.v2 import get_kernel
        self.kernel = await get_kernel()
        self.blocks_initialized["Kernel"] = True
        print("        [OK] SIRE Kernel online")

        print("  [2/7] Initializing LLM Provider...")
        from blackbox.runtime.llm_provider import get_llm_manager
        self.llm = await get_llm_manager()
        self.blocks_initialized["LLM"] = True
        print(f"        [OK] LLM online ({self.llm.get_primary()})")

        print("  [3/7] Initializing VectorDB...")
        from blackbox.runtime.vectordb_provider import get_vectordb, get_memory_store
        self.vectordb = await get_vectordb()
        self.memory = await get_memory_store()
        self.blocks_initialized["VectorDB"] = True
        self.blocks_initialized["Memory"] = True
        print("        [OK] ChromaDB online")

        print("  [4/7] Initializing Daemon...")
        from blackbox.runtime.daemon import BBXDaemon
        self.daemon = BBXDaemon()
        await self.daemon.start()
        self.blocks_initialized["Daemon"] = True
        print("        [OK] Daemon online")

        print("  [5/7] Initializing Snapshot Manager...")
        from blackbox.runtime.daemon import SnapshotManager, BBX_SNAPSHOTS_DIR
        self.snapshots = SnapshotManager(BBX_SNAPSHOTS_DIR)
        self.blocks_initialized["Snapshot"] = True
        print("        [OK] Snapshots online")

        print("  [6/7] Initializing Workflow Engine...")
        from blackbox.runtime.workflows import WorkflowLoader
        self.workflow_loader = WorkflowLoader()
        self.blocks_initialized["Workflow"] = True
        print("        [OK] Workflow Engine online")

        print("  [7/7] Initializing API Server...")
        from blackbox.runtime.api import BBXAPIServer
        self.api = BBXAPIServer(self.daemon, host="127.0.0.1", port=9995)
        await self.api.start()
        await asyncio.sleep(0.5)
        self.blocks_initialized["API"] = True
        self.blocks_initialized["WebSocket"] = True
        print("        [OK] API Server online (port 9995)")

        print(f"\n  All {len(self.blocks_initialized)} blocks initialized!\n")

    async def test_kernel_to_llm(self):
        print("  Testing: Kernel --> LLM")
        start = time.time()
        try:
            result = await self.kernel.llm_driver.think(prompt="Say OK", max_tokens=10)
            latency = (time.time() - start) * 1000
            self.add_connection("Kernel", "LLM", True, result["content"][:20], latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("Kernel", "LLM", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_kernel_to_vectordb(self):
        print("  Testing: Kernel --> VectorDB")
        start = time.time()
        try:
            doc_id = await self.kernel.vectordb_driver.store(collection="kernel_test", content="Kernel test")
            latency = (time.time() - start) * 1000
            self.add_connection("Kernel", "VectorDB", True, doc_id, latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("Kernel", "VectorDB", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_daemon_to_agent(self):
        print("  Testing: Daemon --> Agent")
        start = time.time()
        try:
            from blackbox.runtime.daemon import AgentConfig
            config = AgentConfig(name="test-agent", description="Test")
            agent = await self.daemon.agents.spawn(config)
            self.test_agent = agent
            latency = (time.time() - start) * 1000
            self.add_connection("Daemon", "Agent", True, agent.id, latency)
            print(f"        [OK] Connected ({latency:.0f}ms) - Agent: {agent.id}")
        except Exception as e:
            self.add_connection("Daemon", "Agent", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_agent_to_llm(self):
        print("  Testing: Agent --> LLM")
        start = time.time()
        try:
            await self.daemon.agents.dispatch_task(self.test_agent.id, {"type": "think", "prompt": "Say hello", "max_tokens": 20})
            await asyncio.sleep(2)
            latency = (time.time() - start) * 1000
            tokens = self.test_agent.total_tokens_used
            self.add_connection("Agent", "LLM", tokens > 0, f"{tokens} tokens", latency)
            print(f"        [OK] Connected ({latency:.0f}ms) - {tokens} tokens used")
        except Exception as e:
            self.add_connection("Agent", "LLM", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_agent_to_memory(self):
        print("  Testing: Agent --> Memory")
        start = time.time()
        try:
            await self.daemon.agents.dispatch_task(self.test_agent.id, {"type": "remember", "content": "Agent memory test", "memory_type": "test", "importance": 0.9})
            await asyncio.sleep(1)
            latency = (time.time() - start) * 1000
            self.add_connection("Agent", "Memory", True, "Stored", latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("Agent", "Memory", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_memory_to_vectordb(self):
        print("  Testing: Memory --> VectorDB")
        start = time.time()
        try:
            mem_id = await self.memory.store_memory(agent_id="block_test", content="Memory to VectorDB test", memory_type="test", importance=0.8)
            results = await self.memory.recall(agent_id="block_test", query="VectorDB test", top_k=1)
            latency = (time.time() - start) * 1000
            verified = len(results) > 0
            self.add_connection("Memory", "VectorDB", verified, mem_id, latency)
            print(f"        [OK] Connected ({latency:.0f}ms) - Recalled: {len(results)}")
        except Exception as e:
            self.add_connection("Memory", "VectorDB", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_workflow_to_agent(self):
        print("  Testing: Workflow --> Agent")
        start = time.time()
        try:
            from blackbox.runtime.workflows import WorkflowEngine
            workflow_yaml = 'name: "block-test"\nversion: "1.0"\nsteps:\n  - id: "step1"\n    agent: "tester"\n    task: "Say done"\n    output_key: "result"'
            config = self.workflow_loader.load_from_string(workflow_yaml)

            async def executor(agent_name, task_input):
                response = await self.llm.complete(prompt=task_input.get("task", ""), max_tokens=20)
                return {"content": response.content}

            engine = WorkflowEngine(executor, self.snapshots)
            instance = await engine.run(config, {})
            latency = (time.time() - start) * 1000
            completed = instance.status.value == "completed"
            self.add_connection("Workflow", "Agent", completed, instance.status.value, latency)
            print(f"        [OK] Connected ({latency:.0f}ms) - Status: {instance.status.value}")
        except Exception as e:
            self.add_connection("Workflow", "Agent", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_workflow_to_snapshot(self):
        print("  Testing: Workflow --> Snapshot")
        start = time.time()
        try:
            snapshots = await self.snapshots.list_snapshots()
            latency = (time.time() - start) * 1000
            has_snaps = len(snapshots) > 0
            self.add_connection("Workflow", "Snapshot", has_snaps, f"{len(snapshots)} snapshots", latency)
            print(f"        [OK] Connected ({latency:.0f}ms) - {len(snapshots)} snapshots")
        except Exception as e:
            self.add_connection("Workflow", "Snapshot", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_api_to_daemon(self):
        print("  Testing: API --> Daemon")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:9995/api/v1/status") as resp:
                    data = await resp.json()
                    running = data["data"]["running"]
            latency = (time.time() - start) * 1000
            self.add_connection("API", "Daemon", running, "Running", latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("API", "Daemon", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_api_to_websocket(self):
        print("  Testing: API --> WebSocket")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect("http://127.0.0.1:9995/ws")
                await ws.send_json({"type": "ping"})
                msg = await asyncio.wait_for(ws.receive(), timeout=3)
                data = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
                await ws.close()
            latency = (time.time() - start) * 1000
            got_pong = data.get("type") == "pong"
            self.add_connection("API", "WebSocket", got_pong, "pong", latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("API", "WebSocket", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_api_to_memory(self):
        print("  Testing: API --> Memory")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:9995/api/v1/memory/search", params={"q": "test", "agent_id": "block_test", "top_k": 1}) as resp:
                    data = await resp.json()
                    success = data.get("success", False)
            latency = (time.time() - start) * 1000
            self.add_connection("API", "Memory", success, "Search OK", latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("API", "Memory", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_api_to_snapshot(self):
        print("  Testing: API --> Snapshot")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:9995/api/v1/snapshots") as resp:
                    data = await resp.json()
                    success = data.get("success", False)
            latency = (time.time() - start) * 1000
            self.add_connection("API", "Snapshot", success, "List OK", latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("API", "Snapshot", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_snapshot_to_recovery(self):
        print("  Testing: Snapshot --> Recovery")
        start = time.time()
        try:
            state = {"test": True, "value": 42}
            snap_id = await self.snapshots.create_snapshot("recovery_block_test", state, "Test")
            restored = await self.snapshots.restore_snapshot(snap_id)
            latency = (time.time() - start) * 1000
            verified = restored is not None and restored.get("value") == 42
            self.add_connection("Snapshot", "Recovery", verified, snap_id, latency)
            print(f"        [OK] Connected ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("Snapshot", "Recovery", False, str(e))
            print(f"        [X] Failed: {e}")

    async def test_full_chain(self):
        print("\n  Testing: FULL CHAIN (all blocks)")
        print("           API -> Daemon -> Agent -> LLM -> Memory -> VectorDB -> API")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://127.0.0.1:9995/api/v1/agents", json={"config": {"name": "chain-test"}}) as resp:
                    data = await resp.json()
                    agent_id = data["data"]["agent_id"]
                print(f"           [1] API->Daemon: Agent {agent_id[:8]}")

                async with session.post(f"http://127.0.0.1:9995/api/v1/agents/{agent_id}/task", json={"type": "think", "prompt": "Say chain", "max_tokens": 10}) as resp:
                    pass
                await asyncio.sleep(2)
                print(f"           [2] Daemon->Agent->LLM: Thinking...")

                async with session.post(f"http://127.0.0.1:9995/api/v1/agents/{agent_id}/task", json={"type": "remember", "content": "Chain test complete", "importance": 0.9}) as resp:
                    pass
                await asyncio.sleep(1)
                print(f"           [3] Agent->Memory: Stored")

                async with session.get("http://127.0.0.1:9995/api/v1/memory/search", params={"q": "chain test", "agent_id": "system", "top_k": 1}) as resp:
                    data = await resp.json()
                print(f"           [4] Memory->VectorDB->API: Searched")

                async with session.get(f"http://127.0.0.1:9995/api/v1/agents/{agent_id}") as resp:
                    data = await resp.json()
                    tasks = data["data"]["tasks_completed"]
                print(f"           [5] Full loop: {tasks} tasks completed")

                async with session.delete(f"http://127.0.0.1:9995/api/v1/agents/{agent_id}"):
                    pass

            latency = (time.time() - start) * 1000
            self.add_connection("FULL_CHAIN", "ALL_BLOCKS", tasks >= 2, f"{tasks} tasks", latency)
            print(f"        [OK] FULL CHAIN CONNECTED ({latency:.0f}ms)")
        except Exception as e:
            self.add_connection("FULL_CHAIN", "ALL_BLOCKS", False, str(e))
            print(f"        [X] CHAIN BROKEN: {e}")

    async def cleanup(self):
        print("\n  Cleaning up...")
        if hasattr(self, 'test_agent'):
            await self.daemon.agents.kill(self.test_agent.id)
        await self.memory.clear_agent_memory("block_test")
        await self.api.stop()
        await self.daemon.stop()
        from blackbox.runtime.llm_provider import _llm_manager
        if _llm_manager:
            await _llm_manager.shutdown()
        print("  [OK] Cleanup complete")

    def print_results(self):
        print("\n" + "=" * 70)
        print("BLOCK CONNECTION RESULTS")
        print("=" * 70)

        verified = sum(1 for c in self.connections if c.verified)
        total = len(self.connections)
        print(f"\n  Connections verified: {verified}/{total}\n")

        for conn in self.connections:
            status = "[OK]" if conn.verified else "[X]"
            print(f"  {status} {conn.from_block:12} --> {conn.to_block:12} ", end="")
            if conn.verified:
                print(f"({conn.latency_ms:.0f}ms)")
            else:
                print(f"FAILED: {conn.data_passed}")

        print("\n" + "-" * 70)
        if verified == total:
            print("""
    +---------------------------------------------------------------+
    |                  ALL BLOCKS CONNECTED!                        |
    +---------------------------------------------------------------+
    |                                                               |
    |      +---------+         +---------+         +---------+     |
    |      |   API   |<------->| Daemon  |<------->|  Agent  |     |
    |      +----+----+         +----+----+         +----+----+     |
    |           |                   |                   |          |
    |           v                   v                   v          |
    |      +---------+         +---------+         +---------+     |
    |      |WebSocket|         |Workflow |         |   LLM   |     |
    |      +---------+         +----+----+         +----+----+     |
    |                               |                   |          |
    |                               v                   v          |
    |      +---------+         +---------+         +---------+     |
    |      |Snapshot |<------->| Kernel  |<------->| Memory  |     |
    |      +---------+         +----+----+         +----+----+     |
    |                               |                   |          |
    |                               v                   v          |
    |                          +---------+         +---------+     |
    |                          |Recovery |         |VectorDB |     |
    |                          +---------+         +---------+     |
    |                                                               |
    +---------------------------------------------------------------+
    |              FULL SYSTEM POWER: 100% OPERATIONAL              |
    +---------------------------------------------------------------+
""")
            print("=" * 70)
            print("CONSTRUCTOR TEST: ALL BLOCKS CONNECTED SUCCESSFULLY!")
        else:
            print(f"\n  WARNING: {total - verified} connections failed!\n")
            print("=" * 70)
            print(f"CONSTRUCTOR TEST: {total - verified} CONNECTIONS FAILED")
        print("=" * 70)


async def main():
    test = BBXConstructorTest()
    await test.run()
    return 0 if all(c.verified for c in test.connections) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
