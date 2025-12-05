#!/usr/bin/env python3
"""
BBX Full System Test - No Mocks, Real Everything

This script tests the complete BBX system with real providers:
- ChromaDB for vector storage (semantic memory)
- Anthropic/OpenAI/Ollama for LLM (thinking)
- SIRE Kernel for orchestration

Usage:
    # With Anthropic
    set ANTHROPIC_API_KEY=your_key
    python test_bbx_full.py

    # With OpenAI
    set OPENAI_API_KEY=your_key
    python test_bbx_full.py

    # With Ollama (start ollama first)
    ollama serve
    python test_bbx_full.py
"""

import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_vectordb():
    """Test ChromaDB VectorDB"""
    print("\n=== Testing ChromaDB VectorDB ===")

    from blackbox.runtime import get_vectordb, Document

    db = await get_vectordb()
    print("[OK] ChromaDB initialized")

    # Store documents
    docs = [
        Document(id="doc1", content="BBX is an operating system for AI agents"),
        Document(id="doc2", content="SIRE stands for Synthetic Intelligence Runtime Environment"),
        Document(id="doc3", content="Agents can have persistent memory using vector databases"),
    ]
    await db.store("test_collection", docs)
    print(f"[OK] Stored {len(docs)} documents")

    # Search
    results = await db.search("test_collection", "What is BBX?", top_k=2)
    print(f"[OK] Search found {len(results)} results")
    for r in results:
        print(f"    - [{r.score:.3f}] {r.content[:50]}...")

    # Cleanup
    await db.delete_collection("test_collection")
    print("[OK] ChromaDB test passed")


async def test_llm():
    """Test LLM Providers"""
    print("\n=== Testing LLM Providers ===")

    from blackbox.runtime import get_llm_manager

    manager = await get_llm_manager()

    providers = manager.get_providers()
    primary = manager.get_primary()

    print(f"Available providers: {providers}")
    print(f"Primary provider: {primary}")

    if not providers:
        print("[WARN] No LLM providers available")
        print("       Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or start Ollama")
        return False

    # Test completion
    print("\n[TEST] Calling LLM...")
    response = await manager.complete(
        prompt="What is 2 + 2? Answer in one word.",
        system="You are a helpful assistant. Be concise.",
        temperature=0.1,
        max_tokens=100,
    )

    print(f"[OK] LLM Response: {response.content}")
    print(f"    Model: {response.model}")
    print(f"    Tokens: {response.usage}")
    print(f"    Latency: {response.latency_ms:.1f}ms")

    print("[OK] LLM test passed")
    return True


async def test_sire_kernel():
    """Test SIRE Kernel"""
    print("\n=== Testing SIRE Kernel ===")

    from blackbox.core.v2 import SIREKernel, get_kernel

    kernel = await get_kernel()
    print(f"[OK] SIRE Kernel booted")

    # Test VectorDB through kernel
    doc_id = await kernel.vectordb_driver.store(
        collection="kernel_memories",
        content="The user likes Python programming"
    )
    print(f"[OK] Stored memory: {doc_id}")

    # Search memories
    results = await kernel.vectordb_driver.search(
        collection="kernel_memories",
        query="programming languages",
        top_k=1
    )
    print(f"[OK] Recalled {len(results)} memories")

    # Test LLM through kernel (if available)
    if kernel.llm_driver._llm_manager and kernel.llm_driver._llm_manager.get_providers():
        print("\n[TEST] Calling LLM through SIRE Kernel...")
        result = await kernel.llm_driver.think(
            prompt="Say 'Hello from SIRE!'",
            system="You are a helpful AI running inside the SIRE kernel.",
            max_tokens=50,
        )
        print(f"[OK] LLM Response: {result['content']}")
        print(f"    Latency: {result['latency_ms']:.1f}ms")
    else:
        print("[SKIP] LLM test (no provider available)")

    # Get stats
    stats = kernel.get_stats()
    print(f"\n[OK] Kernel Stats:")
    print(f"    Uptime: {stats['uptime_s']:.2f}s")
    print(f"    LLM calls: {stats['llm'].get('total_calls', 0)}")
    print(f"    VectorDB: {stats.get('vectordb', {})}")

    print("[OK] SIRE Kernel test passed")


async def test_memory_store():
    """Test Agent Memory Store"""
    print("\n=== Testing Agent Memory Store ===")

    from blackbox.runtime import get_memory_store

    store = await get_memory_store()
    print("[OK] Memory store initialized")

    agent_id = "test_agent_001"

    # Store memories
    m1 = await store.store_memory(
        agent_id=agent_id,
        content="The project uses FastAPI for the backend",
        memory_type="fact",
        importance=0.8,
    )
    m2 = await store.store_memory(
        agent_id=agent_id,
        content="User prefers TypeScript over JavaScript",
        memory_type="preference",
        importance=0.9,
    )
    m3 = await store.store_memory(
        agent_id=agent_id,
        content="Had an error with database connection yesterday",
        memory_type="experience",
        importance=0.5,
    )
    print(f"[OK] Stored 3 memories for agent {agent_id}")

    # Recall memories
    results = await store.recall(
        agent_id=agent_id,
        query="What framework does the project use?",
        top_k=2,
    )
    print(f"[OK] Recalled {len(results)} relevant memories")
    for r in results:
        print(f"    - [{r.score:.3f}] {r.content[:50]}...")

    # Count
    count = await store.get_agent_memory_count(agent_id)
    print(f"[OK] Agent has {count} memories")

    # Cleanup
    await store.clear_agent_memory(agent_id)
    print("[OK] Memory store test passed")


async def test_daemon():
    """Test BBX Daemon"""
    print("\n=== Testing BBX Daemon ===")

    from blackbox.runtime import get_daemon

    daemon = get_daemon()
    status = daemon.get_status()

    print(f"[OK] Daemon status:")
    print(f"    Running: {status['running']}")
    print(f"    Home: {status['home_dir']}")
    print(f"    Agents: {status['agents']}")

    print("[OK] Daemon test passed")


async def main():
    print("=" * 60)
    print("BBX FULL SYSTEM TEST - No Mocks, Real Everything")
    print("=" * 60)

    print("\nEnvironment:")
    print(f"  ANTHROPIC_API_KEY: {'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
    print(f"  OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

    try:
        # Test components
        await test_vectordb()
        llm_ok = await test_llm()
        await test_memory_store()
        await test_sire_kernel()
        await test_daemon()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

        if not llm_ok:
            print("\nNote: LLM tests skipped (no provider)")
            print("To enable LLM:")
            print("  1. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            print("  2. Or start Ollama: ollama serve")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean shutdown - close all sessions
        from blackbox.runtime.llm_provider import _llm_manager
        if _llm_manager:
            await _llm_manager.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
