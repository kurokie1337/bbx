#!/usr/bin/env python3
"""
BBX API Server Test - Test all REST endpoints.

Tests:
1. Health check
2. System status
3. Agent CRUD operations
4. Workflow listing and running
5. Memory/RAG operations
6. Snapshot management
"""

import asyncio
import os
import sys
import time
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_api_server():
    """Test BBX API server with all endpoints"""
    print("\n" + "=" * 60)
    print("BBX API SERVER TEST")
    print("=" * 60)

    import aiohttp
    from blackbox.runtime.daemon import BBXDaemon
    from blackbox.runtime.api import BBXAPIServer

    # Start daemon
    daemon = BBXDaemon()
    await daemon.start()
    print("[OK] Daemon started")

    # Start API server
    api = BBXAPIServer(daemon, host="127.0.0.1", port=9998)
    await api.start()
    print("[OK] API server started on http://127.0.0.1:9998")

    # Wait for server to be ready
    await asyncio.sleep(1)

    base_url = "http://127.0.0.1:9998"
    errors = []

    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        print("\n--- Test: Health Check ---")
        try:
            async with session.get(f"{base_url}/health") as resp:
                data = await resp.json()
                assert resp.status == 200
                assert data.get("status") == "ok"
                print(f"[OK] GET /health -> {data}")
        except Exception as e:
            errors.append(f"Health check: {e}")
            print(f"[FAIL] Health check: {e}")

        # Test 2: System status
        print("\n--- Test: System Status ---")
        try:
            async with session.get(f"{base_url}/api/v1/status") as resp:
                data = await resp.json()
                assert resp.status == 200
                assert data.get("success") is True
                print(f"[OK] GET /api/v1/status -> running: {data['data']['running']}")
        except Exception as e:
            errors.append(f"System status: {e}")
            print(f"[FAIL] System status: {e}")

        # Test 3: List agents (empty initially)
        print("\n--- Test: List Agents ---")
        try:
            async with session.get(f"{base_url}/api/v1/agents") as resp:
                data = await resp.json()
                assert resp.status == 200
                print(f"[OK] GET /api/v1/agents -> {len(data['data'])} agents")
        except Exception as e:
            errors.append(f"List agents: {e}")
            print(f"[FAIL] List agents: {e}")

        # Test 4: Spawn agent
        print("\n--- Test: Spawn Agent ---")
        agent_id = None
        try:
            async with session.post(
                f"{base_url}/api/v1/agents",
                json={
                    "config": {
                        "name": "api-test-agent",
                        "model": "qwen2.5:0.5b",
                        "description": "Test agent from API",
                    }
                }
            ) as resp:
                data = await resp.json()
                assert resp.status == 200
                assert data.get("success") is True
                agent_id = data["data"]["agent_id"]
                print(f"[OK] POST /api/v1/agents -> agent_id: {agent_id}")
        except Exception as e:
            errors.append(f"Spawn agent: {e}")
            print(f"[FAIL] Spawn agent: {e}")

        # Wait for agent to start
        await asyncio.sleep(0.5)

        # Test 5: Get agent details
        print("\n--- Test: Get Agent Details ---")
        if agent_id:
            try:
                async with session.get(f"{base_url}/api/v1/agents/{agent_id}") as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] GET /api/v1/agents/{agent_id} -> status: {data['data']['status']}")
            except Exception as e:
                errors.append(f"Get agent: {e}")
                print(f"[FAIL] Get agent: {e}")

        # Test 6: Dispatch task to agent
        print("\n--- Test: Dispatch Task ---")
        if agent_id:
            try:
                async with session.post(
                    f"{base_url}/api/v1/agents/{agent_id}/task",
                    json={
                        "type": "think",
                        "prompt": "Say hello",
                        "max_tokens": 20,
                    }
                ) as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] POST /api/v1/agents/{agent_id}/task -> dispatched")
            except Exception as e:
                errors.append(f"Dispatch task: {e}")
                print(f"[FAIL] Dispatch task: {e}")

        # Wait for task
        await asyncio.sleep(2)

        # Test 7: Pause agent
        print("\n--- Test: Pause Agent ---")
        if agent_id:
            try:
                async with session.post(f"{base_url}/api/v1/agents/{agent_id}/pause") as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] POST /api/v1/agents/{agent_id}/pause -> success: {data['success']}")
            except Exception as e:
                errors.append(f"Pause agent: {e}")
                print(f"[FAIL] Pause agent: {e}")

        # Test 8: Resume agent
        print("\n--- Test: Resume Agent ---")
        if agent_id:
            try:
                async with session.post(f"{base_url}/api/v1/agents/{agent_id}/resume") as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] POST /api/v1/agents/{agent_id}/resume -> success: {data['success']}")
            except Exception as e:
                errors.append(f"Resume agent: {e}")
                print(f"[FAIL] Resume agent: {e}")

        # Test 9: List workflows
        print("\n--- Test: List Workflows ---")
        try:
            async with session.get(f"{base_url}/api/v1/workflows") as resp:
                data = await resp.json()
                assert resp.status == 200
                workflows = data.get("data", [])
                print(f"[OK] GET /api/v1/workflows -> {len(workflows)} workflows")
                for w in workflows[:3]:
                    print(f"      - {w['name']}: {w['steps_count']} steps")
        except Exception as e:
            errors.append(f"List workflows: {e}")
            print(f"[FAIL] List workflows: {e}")

        # Test 10: Memory stats
        print("\n--- Test: Memory Stats ---")
        try:
            async with session.get(f"{base_url}/api/v1/memory") as resp:
                data = await resp.json()
                assert resp.status == 200
                print(f"[OK] GET /api/v1/memory -> {data['data']}")
        except Exception as e:
            errors.append(f"Memory stats: {e}")
            print(f"[FAIL] Memory stats: {e}")

        # Test 11: Memory search (RAG)
        print("\n--- Test: Memory Search (RAG) ---")
        try:
            async with session.get(
                f"{base_url}/api/v1/memory/search",
                params={"q": "programming", "agent_id": "test", "top_k": 3}
            ) as resp:
                data = await resp.json()
                assert resp.status == 200
                results = data.get("data", [])
                print(f"[OK] GET /api/v1/memory/search -> {len(results)} results")
        except Exception as e:
            errors.append(f"Memory search: {e}")
            print(f"[FAIL] Memory search: {e}")

        # Test 12: List snapshots
        print("\n--- Test: List Snapshots ---")
        try:
            async with session.get(f"{base_url}/api/v1/snapshots") as resp:
                data = await resp.json()
                assert resp.status == 200
                snapshots = data.get("data", [])
                print(f"[OK] GET /api/v1/snapshots -> {len(snapshots)} snapshots")
        except Exception as e:
            errors.append(f"List snapshots: {e}")
            print(f"[FAIL] List snapshots: {e}")

        # Test 13: Create snapshot
        print("\n--- Test: Create Snapshot ---")
        snapshot_id = None
        try:
            async with session.post(
                f"{base_url}/api/v1/snapshots",
                json={
                    "agent_id": "api_test",
                    "description": "API test snapshot",
                    "state": {"test": True, "value": 42}
                }
            ) as resp:
                data = await resp.json()
                assert resp.status == 200
                snapshot_id = data["data"]["snapshot_id"]
                print(f"[OK] POST /api/v1/snapshots -> {snapshot_id}")
        except Exception as e:
            errors.append(f"Create snapshot: {e}")
            print(f"[FAIL] Create snapshot: {e}")

        # Test 14: Recover snapshot
        print("\n--- Test: Recover Snapshot ---")
        if snapshot_id:
            try:
                async with session.post(f"{base_url}/api/v1/snapshots/{snapshot_id}/recover") as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] POST /api/v1/snapshots/{snapshot_id}/recover -> success: {data['success']}")
            except Exception as e:
                errors.append(f"Recover snapshot: {e}")
                print(f"[FAIL] Recover snapshot: {e}")

        # Test 15: Get logs
        print("\n--- Test: Get Logs ---")
        try:
            async with session.get(f"{base_url}/api/v1/logs") as resp:
                data = await resp.json()
                assert resp.status == 200
                logs = data.get("data", [])
                print(f"[OK] GET /api/v1/logs -> {len(logs)} log entries")
        except Exception as e:
            errors.append(f"Get logs: {e}")
            print(f"[FAIL] Get logs: {e}")

        # Test 16: Kill agent
        print("\n--- Test: Kill Agent ---")
        if agent_id:
            try:
                async with session.delete(f"{base_url}/api/v1/agents/{agent_id}") as resp:
                    data = await resp.json()
                    assert resp.status == 200
                    print(f"[OK] DELETE /api/v1/agents/{agent_id} -> success: {data['success']}")
            except Exception as e:
                errors.append(f"Kill agent: {e}")
                print(f"[FAIL] Kill agent: {e}")

    # Stop servers
    await api.stop()
    await daemon.stop()
    print("\n[OK] Servers stopped")

    # Summary
    print("\n" + "=" * 60)
    print("API TEST RESULTS")
    print("=" * 60)

    if errors:
        print(f"\n[WARN] {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n[PASS] All API tests passed!")
        return True


async def main():
    print("=" * 60)
    print("BBX API SERVER TEST SUITE")
    print("=" * 60)

    try:
        result = await test_api_server()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        result = False

    # Cleanup
    from blackbox.runtime.llm_provider import _llm_manager
    if _llm_manager:
        await _llm_manager.shutdown()

    if result:
        print("\n" + "=" * 60)
        print("ALL API TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
