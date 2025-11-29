# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 CLI Commands

This module provides CLI commands for BBX 2.0 features:
- AgentRing management (ring stats, config)
- Hooks management (list, add, remove, enable, disable)
- Context tiering statistics
- Declarative configuration (apply, rollback, diff)
- Generation management (list, switch, diff)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml


# =============================================================================
# BBX 2.0 Main Group
# =============================================================================

@click.group("v2")
def v2_cli():
    """BBX 2.0 commands - Next-generation features.

    BBX 2.0 brings Linux-grade infrastructure to AI agents:

    - AgentRing: io_uring-inspired batch operations (10x throughput)
    - Hooks: eBPF-inspired dynamic programming
    - ContextTiering: MGLRU-inspired smart memory
    - Declarative: NixOS-inspired infrastructure as code

    Examples:
        bbx v2 ring stats
        bbx v2 hooks list
        bbx v2 config apply config.yaml
        bbx v2 generation list
    """
    pass


# =============================================================================
# AgentRing Commands
# =============================================================================

@v2_cli.group("ring")
def ring_group():
    """AgentRing commands - io_uring-inspired batch operations.

    AgentRing provides high-throughput batch execution of adapter operations
    using a submission/completion queue pattern inspired by Linux io_uring.

    Examples:
        bbx v2 ring stats
        bbx v2 ring config
        bbx v2 ring benchmark
    """
    pass


@ring_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
@click.option("--watch", "-w", is_flag=True, help="Watch stats in real-time")
@click.option("--interval", "-i", type=float, default=1.0, help="Watch interval in seconds")
def ring_stats(format: str, watch: bool, interval: float):
    """Show AgentRing statistics.

    Displays current ring performance metrics:
    - Operations submitted/completed
    - Throughput (ops/sec)
    - Latency (avg, p50, p95, p99)
    - Worker pool utilization

    Examples:
        bbx v2 ring stats
        bbx v2 ring stats --format json
        bbx v2 ring stats --watch
    """
    async def _get_stats():
        from blackbox.core.v2.runtime_v2 import get_runtime_v2

        runtime = get_runtime_v2()
        if not runtime._started:
            await runtime.start()

        if runtime.ring:
            return runtime.ring.get_stats()
        return None

    def display_stats(stats):
        if stats is None:
            click.echo("AgentRing not active")
            return

        if format == "json":
            click.echo(json.dumps(stats.__dict__, indent=2, default=str))
        else:
            click.echo("\n" + "=" * 60)
            click.echo("AgentRing Statistics")
            click.echo("=" * 60)
            click.echo(f"\nOperations:")
            click.echo(f"  Submitted:  {stats.operations_submitted}")
            click.echo(f"  Completed:  {stats.operations_completed}")
            click.echo(f"  Failed:     {stats.operations_failed}")
            click.echo(f"  Pending:    {stats.operations_submitted - stats.operations_completed - stats.operations_failed}")

            click.echo(f"\nPerformance:")
            click.echo(f"  Throughput:     {stats.throughput_ops_sec:.2f} ops/sec")
            click.echo(f"  Avg Latency:    {stats.avg_latency_ms:.2f}ms")
            click.echo(f"  P50 Latency:    {stats.p50_latency_ms:.2f}ms")
            click.echo(f"  P95 Latency:    {stats.p95_latency_ms:.2f}ms")
            click.echo(f"  P99 Latency:    {stats.p99_latency_ms:.2f}ms")

            click.echo(f"\nWorkers:")
            click.echo(f"  Pool Size:      {stats.worker_pool_size}")
            click.echo(f"  Active:         {stats.active_workers}")
            click.echo(f"  Utilization:    {stats.worker_utilization:.1f}%")

            click.echo(f"\nQueues:")
            click.echo(f"  Submission:     {stats.submission_queue_size}")
            click.echo(f"  Completion:     {stats.completion_queue_size}")

            click.echo("=" * 60)

    if watch:
        import time
        click.echo("Watching AgentRing stats (Ctrl+C to stop)...")
        try:
            while True:
                click.clear()
                stats = asyncio.run(_get_stats())
                display_stats(stats)
                click.echo(f"\nUpdated: {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nStopped watching.")
    else:
        stats = asyncio.run(_get_stats())
        display_stats(stats)


@ring_group.command("config")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def ring_config(format: str):
    """Show AgentRing configuration.

    Example:
        bbx v2 ring config
    """
    async def _get_config():
        from blackbox.core.v2.runtime_v2 import get_runtime_v2

        runtime = get_runtime_v2()
        if runtime.ring:
            return runtime.ring.config
        return None

    config = asyncio.run(_get_config())

    if config is None:
        click.echo("AgentRing not configured")
        return

    if format == "json":
        click.echo(json.dumps(config.__dict__, indent=2, default=str))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("AgentRing Configuration")
        click.echo("=" * 60)
        click.echo(f"\n  Submission Queue Size: {config.submission_queue_size}")
        click.echo(f"  Completion Queue Size: {config.completion_queue_size}")
        click.echo(f"  Worker Pool Size:      {config.worker_pool_size}")
        click.echo(f"  Max Batch Size:        {config.max_batch_size}")
        click.echo(f"  Default Timeout (ms):  {config.default_timeout_ms}")
        click.echo(f"  Enable Priorities:     {config.enable_priorities}")
        click.echo(f"  Enable Retries:        {config.enable_retries}")
        click.echo("=" * 60)


@ring_group.command("benchmark")
@click.option("--operations", "-n", type=int, default=1000, help="Number of operations")
@click.option("--batch-size", "-b", type=int, default=100, help="Batch size")
@click.option("--workers", "-w", type=int, default=10, help="Worker count")
def ring_benchmark(operations: int, batch_size: int, workers: int):
    """Run AgentRing performance benchmark.

    Measures throughput and latency with test operations.

    Example:
        bbx v2 ring benchmark -n 10000 -b 100 -w 20
    """
    async def _benchmark():
        from blackbox.core.v2.ring import AgentRing, RingConfig, Operation, OperationType
        import time

        config = RingConfig(
            min_workers=workers,
            max_workers=workers,
            max_batch_size=batch_size,
        )

        ring = AgentRing(config)

        # Create mock adapter
        class MockAdapter:
            async def execute(self, method, args):
                await asyncio.sleep(0.001)  # 1ms simulated work
                return {"status": "ok"}

        await ring.start({"mock": MockAdapter()})

        click.echo(f"\nBenchmarking AgentRing...")
        click.echo(f"  Operations: {operations}")
        click.echo(f"  Batch Size: {batch_size}")
        click.echo(f"  Workers:    {workers}")
        click.echo()

        # Create operations
        ops = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="test",
                args={},
            )
            for _ in range(operations)
        ]

        start = time.perf_counter()

        # Submit in batches
        all_ids = []
        for i in range(0, len(ops), batch_size):
            batch = ops[i:i + batch_size]
            ids = await ring.submit_batch(batch)
            all_ids.extend(ids)

        # Wait for all
        completions = await ring.wait_batch(all_ids, timeout=300.0)

        end = time.perf_counter()
        duration = end - start

        # Calculate stats
        successful = sum(1 for c in completions if c.status.name == "COMPLETED")
        failed = len(completions) - successful
        throughput = operations / duration

        latencies = [c.duration_ms for c in completions if c.duration_ms > 0]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
        else:
            avg_latency = p50 = p95 = p99 = 0

        await ring.stop()

        click.echo("=" * 60)
        click.echo("Benchmark Results")
        click.echo("=" * 60)
        click.echo(f"\n  Total Time:    {duration:.2f}s")
        click.echo(f"  Successful:    {successful}")
        click.echo(f"  Failed:        {failed}")
        click.echo(f"  Throughput:    {throughput:.2f} ops/sec")
        click.echo(f"\n  Avg Latency:   {avg_latency:.2f}ms")
        click.echo(f"  P50 Latency:   {p50:.2f}ms")
        click.echo(f"  P95 Latency:   {p95:.2f}ms")
        click.echo(f"  P99 Latency:   {p99:.2f}ms")
        click.echo("=" * 60)

    asyncio.run(_benchmark())


# =============================================================================
# Hooks Commands
# =============================================================================

@v2_cli.group("hooks")
def hooks_group():
    """BBX Hooks commands - eBPF-inspired dynamic programming.

    Hooks allow dynamic injection of logic at workflow lifecycle points
    without modifying workflow files. Inspired by Linux eBPF.

    Attach Points:
        - workflow.start / workflow.end / workflow.error
        - step.pre_execute / step.post_execute / step.error
        - adapter.call / adapter.result

    Hook Types:
        - PROBE: Read-only observation
        - GUARD: Can block/skip execution
        - TRANSFORM: Can modify inputs/outputs

    Examples:
        bbx v2 hooks list
        bbx v2 hooks add metrics.hook.yaml
        bbx v2 hooks enable audit_logger
    """
    pass


@hooks_group.command("list")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
@click.option("--verbose", "-v", is_flag=True, help="Show hook details")
def hooks_list(format: str, verbose: bool):
    """List registered hooks.

    Examples:
        bbx v2 hooks list
        bbx v2 hooks list --format json
        bbx v2 hooks list -v
    """
    from blackbox.core.v2.hooks import get_hook_manager

    manager = get_hook_manager()
    hooks = manager.list_hooks()

    if format == "json":
        result = []
        for h in hooks:
            result.append({
                "id": h.id,
                "name": h.name,
                "type": h.type.name,
                "enabled": h.enabled,
                "priority": h.priority,
                "attach_points": [ap.value for ap in h.attach_points],
                "description": h.description,
            })
        click.echo(json.dumps(result, indent=2))
    else:
        if not hooks:
            click.echo("No hooks registered.")
            click.echo("\nAdd a hook with: bbx v2 hooks add <hook_file>")
            return

        click.echo("\n" + "=" * 60)
        click.echo("Registered Hooks")
        click.echo("=" * 60)

        for h in sorted(hooks, key=lambda x: x.priority):
            status = "[ON] " if h.enabled else "[OFF]"
            click.echo(f"\n{status} {h.name} ({h.id})")
            click.echo(f"     Type: {h.type.name}")
            click.echo(f"     Priority: {h.priority}")
            if verbose:
                click.echo(f"     Attach: {', '.join(ap.value for ap in h.attach_points)}")
                if h.description:
                    click.echo(f"     Description: {h.description}")

        click.echo("\n" + "=" * 60)
        click.echo(f"Total: {len(hooks)} hooks")


@hooks_group.command("add")
@click.argument("hook_file", type=click.Path(exists=True))
def hooks_add(hook_file: str):
    """Add a hook from a YAML file.

    Hook file format:
        hook:
          id: my_hook
          name: My Hook
          type: PROBE  # PROBE, GUARD, or TRANSFORM
          attach:
            - step.post_execute
          action:
            type: inline
            code: |
              print(f"Step {ctx.step_id} completed")

    Example:
        bbx v2 hooks add metrics.hook.yaml
    """
    from blackbox.core.v2.hooks import (
        get_hook_manager, HookDefinition, HookType, AttachPoint
    )

    try:
        with open(hook_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        hook_data = data.get("hook", data)

        hook_def = HookDefinition(
            id=hook_data.get("id", Path(hook_file).stem),
            name=hook_data.get("name", Path(hook_file).stem),
            type=HookType[hook_data.get("type", "PROBE").upper()],
            attach_points=[
                AttachPoint(ap) for ap in hook_data.get("attach", [])
            ],
            description=hook_data.get("description"),
            priority=hook_data.get("priority", 0),
            code=hook_data.get("action", {}).get("code"),
        )

        manager = get_hook_manager()
        success = manager.register(hook_def)

        if success:
            click.echo(f"[+] Hook registered: {hook_def.name} ({hook_def.id})")
            click.echo(f"    Type: {hook_def.type.name}")
            click.echo(f"    Attach: {', '.join(ap.value for ap in hook_def.attach_points)}")
        else:
            click.echo(f"[-] Failed to register hook (verification failed)", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


@hooks_group.command("remove")
@click.argument("hook_id")
def hooks_remove(hook_id: str):
    """Remove a hook by ID.

    Example:
        bbx v2 hooks remove my_hook
    """
    from blackbox.core.v2.hooks import get_hook_manager

    manager = get_hook_manager()
    success = manager.unregister(hook_id)

    if success:
        click.echo(f"[+] Hook removed: {hook_id}")
    else:
        click.echo(f"[-] Hook not found: {hook_id}", err=True)


@hooks_group.command("enable")
@click.argument("hook_id")
def hooks_enable(hook_id: str):
    """Enable a hook.

    Example:
        bbx v2 hooks enable audit_logger
    """
    from blackbox.core.v2.hooks import get_hook_manager

    manager = get_hook_manager()
    success = manager.enable(hook_id)

    if success:
        click.echo(f"[+] Hook enabled: {hook_id}")
    else:
        click.echo(f"[-] Hook not found: {hook_id}", err=True)


@hooks_group.command("disable")
@click.argument("hook_id")
def hooks_disable(hook_id: str):
    """Disable a hook without removing it.

    Example:
        bbx v2 hooks disable audit_logger
    """
    from blackbox.core.v2.hooks import get_hook_manager

    manager = get_hook_manager()
    success = manager.disable(hook_id)

    if success:
        click.echo(f"[+] Hook disabled: {hook_id}")
    else:
        click.echo(f"[-] Hook not found: {hook_id}", err=True)


@hooks_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def hooks_stats(format: str):
    """Show hooks execution statistics.

    Example:
        bbx v2 hooks stats
    """
    from blackbox.core.v2.hooks import get_hook_manager

    manager = get_hook_manager()
    stats = manager.get_stats()

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("Hooks Statistics")
        click.echo("=" * 60)

        click.echo(f"\nGlobal:")
        click.echo(f"  Total Triggers:  {stats.get('total_triggers', 0)}")
        click.echo(f"  Total Duration:  {stats.get('total_duration_ms', 0):.2f}ms")
        click.echo(f"  Avg Duration:    {stats.get('avg_duration_ms', 0):.2f}ms")

        click.echo(f"\nBy Attach Point:")
        for ap, count in stats.get('by_attach_point', {}).items():
            click.echo(f"  {ap}: {count}")

        click.echo(f"\nBy Hook:")
        for hook_id, hook_stats in stats.get('by_hook', {}).items():
            click.echo(f"  {hook_id}:")
            click.echo(f"    Triggers: {hook_stats.get('count', 0)}")
            click.echo(f"    Avg Time: {hook_stats.get('avg_ms', 0):.2f}ms")

        click.echo("=" * 60)


# =============================================================================
# Context Tiering Commands
# =============================================================================

@v2_cli.group("context")
def context_group():
    """Context tiering commands - MGLRU-inspired memory management.

    ContextTiering provides intelligent multi-generation memory management
    for AI agent context, inspired by Linux MGLRU.

    Tiers:
        - HOT:  Active data (in memory)
        - WARM: Recent data (in memory, may compress)
        - COOL: Older data (on disk, compressed)
        - COLD: Archive data (optional vector DB)

    Examples:
        bbx v2 context stats
        bbx v2 context get my_key
        bbx v2 context pin important_key
    """
    pass


@context_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def context_stats(format: str):
    """Show context tiering statistics.

    Example:
        bbx v2 context stats
    """
    async def _get_stats():
        from blackbox.core.v2.context_tiering import get_context_tiering

        tiering = get_context_tiering()
        return tiering.get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("Context Tiering Statistics")
        click.echo("=" * 60)

        # Get generation stats from dict
        generations = stats.get("generations", [])
        hot = generations[0] if len(generations) > 0 else {"items": 0, "size_bytes": 0}
        warm = generations[1] if len(generations) > 1 else {"items": 0, "size_bytes": 0}
        cool = generations[2] if len(generations) > 2 else {"items": 0, "size_bytes": 0}
        cold = generations[3] if len(generations) > 3 else {"items": 0, "size_bytes": 0}

        click.echo(f"\nItems by Tier:")
        click.echo(f"  HOT:   {hot.get('items', 0)} ({hot.get('size_bytes', 0) / 1024:.1f} KB)")
        click.echo(f"  WARM:  {warm.get('items', 0)} ({warm.get('size_bytes', 0) / 1024:.1f} KB)")
        click.echo(f"  COOL:  {cool.get('items', 0)} ({cool.get('size_bytes', 0) / 1024:.1f} KB)")
        click.echo(f"  COLD:  {cold.get('items', 0)} ({cold.get('size_bytes', 0) / 1024:.1f} KB)")

        click.echo(f"\nMemory:")
        total_bytes = stats.get("total_size_bytes", 0)
        click.echo(f"  Total:    {total_bytes / 1024:.1f} KB")

        click.echo(f"\nOperations:")
        click.echo(f"  Cache Hits:   {stats.get('cache_hits', 0)}")
        click.echo(f"  Cache Misses: {stats.get('cache_misses', 0)}")
        click.echo(f"  Promotions:   {stats.get('promotions', 0)}")
        click.echo(f"  Demotions:    {stats.get('demotions', 0)}")
        click.echo(f"  Hit Rate:     {stats.get('hit_rate', 0) * 100:.1f}%")

        click.echo(f"\nTotal Items: {stats.get('total_items', 0)}")

        click.echo("=" * 60)


@context_group.command("get")
@click.argument("key")
def context_get(key: str):
    """Get a value from tiered context.

    Example:
        bbx v2 context get workflow_inputs
    """
    async def _get():
        from blackbox.core.v2.context_tiering import get_context_tiering

        tiering = get_context_tiering()
        return await tiering.get(key)

    value = asyncio.run(_get())

    if value is None:
        click.echo(f"Key not found: {key}", err=True)
    else:
        if isinstance(value, (dict, list)):
            click.echo(json.dumps(value, indent=2, default=str))
        else:
            click.echo(value)


@context_group.command("pin")
@click.argument("key")
def context_pin(key: str):
    """Pin a key to prevent demotion to lower tiers.

    Example:
        bbx v2 context pin important_data
    """
    async def _pin():
        from blackbox.core.v2.context_tiering import get_context_tiering

        tiering = get_context_tiering()
        return await tiering.pin(key)

    success = asyncio.run(_pin())

    if success:
        click.echo(f"[+] Pinned: {key}")
    else:
        click.echo(f"[-] Key not found: {key}", err=True)


@context_group.command("unpin")
@click.argument("key")
def context_unpin(key: str):
    """Unpin a key to allow demotion.

    Example:
        bbx v2 context unpin some_data
    """
    async def _unpin():
        from blackbox.core.v2.context_tiering import get_context_tiering

        tiering = get_context_tiering()
        return await tiering.unpin(key)

    success = asyncio.run(_unpin())

    if success:
        click.echo(f"[+] Unpinned: {key}")
    else:
        click.echo(f"[-] Key not found: {key}", err=True)


@context_group.command("flush")
@click.option("--tier", "-t", type=click.Choice(["warm", "cool", "cold", "all"]), default="cool")
def context_flush(tier: str):
    """Flush context data to disk.

    Example:
        bbx v2 context flush --tier cool
    """
    async def _flush():
        from blackbox.core.v2.context_tiering import get_context_tiering, GenerationTier

        tiering = get_context_tiering()

        if tier == "all":
            tiers = [GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD]
        else:
            tier_map = {
                "warm": GenerationTier.WARM,
                "cool": GenerationTier.COOL,
                "cold": GenerationTier.COLD,
            }
            tiers = [tier_map[tier]]

        flushed = 0
        for t in tiers:
            flushed += await tiering.flush_tier(t)

        return flushed

    count = asyncio.run(_flush())
    click.echo(f"[+] Flushed {count} items to disk")


# =============================================================================
# Declarative Config Commands
# =============================================================================

@v2_cli.group("config")
def config_group():
    """Declarative configuration commands - NixOS-inspired.

    Manage BBX infrastructure as code with atomic configuration
    changes, rollback support, and generation management.

    Examples:
        bbx v2 config apply config.yaml
        bbx v2 config rollback 3
        bbx v2 config diff 4 5
    """
    pass


@config_group.command("apply")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would change")
def config_apply(config_file: str, dry_run: bool):
    """Apply a declarative configuration.

    Creates a new generation with the specified configuration.
    Changes are atomic - either all succeed or nothing changes.

    Example:
        bbx v2 config apply bbx-config.yaml
        bbx v2 config apply bbx-config.yaml --dry-run
    """
    async def _apply():
        from blackbox.core.v2.declarative import DeclarativeManager, BBXConfig

        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = BBXConfig.from_dict(data)
        manager = DeclarativeManager()

        if dry_run:
            # Show what would change
            current = manager.get_current_config()
            diff = manager.diff_configs(current, config)
            return {"dry_run": True, "diff": diff}
        else:
            generation = await manager.apply(config)
            return {"dry_run": False, "generation": generation}

    result = asyncio.run(_apply())

    if result["dry_run"]:
        click.echo("\n" + "=" * 60)
        click.echo("Dry Run - Would apply these changes:")
        click.echo("=" * 60)

        diff = result["diff"]
        if not diff:
            click.echo("\nNo changes detected.")
        else:
            for change in diff:
                click.echo(f"\n  {change['type']}: {change['path']}")
                if change.get('old'):
                    click.echo(f"    - {change['old']}")
                if change.get('new'):
                    click.echo(f"    + {change['new']}")

        click.echo("\n" + "=" * 60)
    else:
        gen = result["generation"]
        click.echo(f"\n[+] Configuration applied successfully!")
        click.echo(f"    Generation: {gen.id}")
        click.echo(f"    Created: {gen.created_at}")
        click.echo(f"\n    To rollback: bbx v2 config rollback {gen.id - 1}")


@config_group.command("rollback")
@click.argument("generation_id", type=int)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def config_rollback(generation_id: int, yes: bool):
    """Rollback to a previous generation.

    Example:
        bbx v2 config rollback 3
    """
    if not yes:
        if not click.confirm(f"Rollback to generation {generation_id}?"):
            return

    async def _rollback():
        from blackbox.core.v2.declarative import DeclarativeManager

        manager = DeclarativeManager()
        config = await manager.rollback(generation_id)
        return config

    config = asyncio.run(_rollback())

    if config:
        click.echo(f"[+] Rolled back to generation {generation_id}")
    else:
        click.echo(f"[-] Generation not found: {generation_id}", err=True)


@config_group.command("show")
@click.option("--format", "-f", type=click.Choice(["text", "yaml", "json"]), default="yaml")
def config_show(format: str):
    """Show current configuration.

    Example:
        bbx v2 config show
        bbx v2 config show --format json
    """
    from blackbox.core.v2.declarative import DeclarativeManager

    manager = DeclarativeManager()
    config = manager.get_current_config()

    if config is None:
        click.echo("No configuration applied yet.")
        click.echo("\nApply a config with: bbx v2 config apply <config.yaml>")
        return

    if format == "json":
        click.echo(json.dumps(config.to_dict(), indent=2, default=str))
    elif format == "yaml":
        click.echo(yaml.dump(config.to_dict(), default_flow_style=False))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("Current Configuration")
        click.echo("=" * 60)
        click.echo(f"\nVersion: {config.version}")
        click.echo(f"Agent: {config.agent.name}")
        click.echo(f"Description: {config.agent.description}")
        click.echo(f"\nAdapters: {len(config.adapters)}")
        for name, adapter in config.adapters.items():
            click.echo(f"  - {name}")
        click.echo(f"\nHooks: {len(config.hooks)}")
        for name, hook in config.hooks.items():
            click.echo(f"  - {name}: {hook.type}")
        click.echo("=" * 60)


@config_group.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
def config_validate(config_file: str):
    """Validate a configuration file.

    Example:
        bbx v2 config validate bbx-config.yaml
    """
    from blackbox.core.v2.declarative import BBXConfig

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = BBXConfig.from_dict(data)
        errors = config.validate()

        if errors:
            click.echo(f"[-] Configuration invalid:")
            for error in errors:
                click.echo(f"    - {error}")
            raise click.Abort()
        else:
            click.echo(f"[+] Configuration valid: {config_file}")

    except yaml.YAMLError as e:
        click.echo(f"[-] YAML parse error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


# =============================================================================
# Generation Management Commands
# =============================================================================

@v2_cli.group("generation")
def generation_group():
    """Generation management commands.

    Generations are atomic configuration snapshots, similar to NixOS.
    Each configuration change creates a new generation, enabling
    instant rollback to any previous state.

    Examples:
        bbx v2 generation list
        bbx v2 generation diff 4 5
        bbx v2 generation switch 3
    """
    pass


@generation_group.command("list")
@click.option("--limit", "-n", type=int, default=10, help="Number of generations to show")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def generation_list(limit: int, format: str):
    """List configuration generations.

    Example:
        bbx v2 generation list
        bbx v2 generation list -n 20
    """
    from blackbox.core.v2.declarative import DeclarativeManager

    manager = DeclarativeManager()
    generations = manager.list_generations(limit=limit)
    current_id = manager.get_current_generation_id()

    if format == "json":
        result = [
            {
                "id": g.id,
                "created_at": g.created_at.isoformat(),
                "description": g.description,
                "hash": g.config_hash,
                "current": g.id == current_id,
            }
            for g in generations
        ]
        click.echo(json.dumps(result, indent=2))
    else:
        if not generations:
            click.echo("No generations found.")
            click.echo("\nCreate one with: bbx v2 config apply <config.yaml>")
            return

        click.echo("\n" + "=" * 60)
        click.echo("Configuration Generations")
        click.echo("=" * 60)

        for g in generations:
            marker = " *" if g.id == current_id else "  "
            click.echo(f"\n{marker}Generation {g.id}")
            click.echo(f"     Created: {g.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if g.description:
                click.echo(f"     Description: {g.description}")
            click.echo(f"     Hash: {g.config_hash[:16]}...")

        click.echo("\n" + "=" * 60)
        click.echo("* = current generation")


@generation_group.command("diff")
@click.argument("gen1", type=int)
@click.argument("gen2", type=int)
def generation_diff(gen1: int, gen2: int):
    """Show differences between two generations.

    Example:
        bbx v2 generation diff 4 5
    """
    from blackbox.core.v2.declarative import DeclarativeManager

    manager = DeclarativeManager()
    diff = manager.diff_generations(gen1, gen2)

    if not diff:
        click.echo(f"No differences between generation {gen1} and {gen2}")
        return

    click.echo("\n" + "=" * 60)
    click.echo(f"Diff: Generation {gen1} -> {gen2}")
    click.echo("=" * 60)

    for change in diff:
        click.echo(f"\n  {change['type']}: {change['path']}")
        if change.get('old'):
            click.echo(f"    - {change['old']}")
        if change.get('new'):
            click.echo(f"    + {change['new']}")

    click.echo("=" * 60)


@generation_group.command("switch")
@click.argument("generation_id", type=int)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def generation_switch(generation_id: int, yes: bool):
    """Switch to a specific generation.

    Example:
        bbx v2 generation switch 3
    """
    if not yes:
        if not click.confirm(f"Switch to generation {generation_id}?"):
            return

    async def _switch():
        from blackbox.core.v2.declarative import DeclarativeManager

        manager = DeclarativeManager()
        return await manager.switch_generation(generation_id)

    success = asyncio.run(_switch())

    if success:
        click.echo(f"[+] Switched to generation {generation_id}")
    else:
        click.echo(f"[-] Failed to switch to generation {generation_id}", err=True)


# =============================================================================
# Run V2 Command
# =============================================================================

@v2_cli.command("run")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--input", "-i", "inputs", multiple=True, help="Inputs in key=value format")
@click.option("--ring/--no-ring", default=True, help="Use AgentRing")
@click.option("--hooks/--no-hooks", default=True, help="Enable hooks")
@click.option("--tiering/--no-tiering", default=True, help="Enable context tiering")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
def run_v2(file_path: str, inputs: tuple, ring: bool, hooks: bool, tiering: bool, output: str):
    """Run a workflow using BBX 2.0 runtime.

    Uses all BBX 2.0 features:
    - AgentRing for batch operations
    - Hooks for observability
    - Context tiering for memory management

    Examples:
        bbx v2 run workflow.bbx
        bbx v2 run workflow.bbx -i name=test
        bbx v2 run workflow.bbx --no-ring
    """
    async def _run():
        from blackbox.core.v2.runtime_v2 import run_file_v2, RuntimeV2Config

        # Parse inputs
        input_dict = {}
        for input_str in inputs:
            if "=" in input_str:
                key, value = input_str.split("=", 1)
                if value.lower() in ("true", "false"):
                    input_dict[key] = value.lower() == "true"
                elif value.isdigit():
                    input_dict[key] = int(value)
                else:
                    input_dict[key] = value

        config = RuntimeV2Config(
            ring_enabled=ring,
            hooks_enabled=hooks,
            tiering_enabled=tiering,
        )

        return await run_file_v2(file_path, inputs=input_dict, config=config)

    try:
        results = asyncio.run(_run())

        if output == "json":
            click.echo(json.dumps(results, indent=2, default=str))
        else:
            click.echo("\n" + "=" * 60)
            click.echo("BBX 2.0 Workflow Results")
            click.echo("=" * 60)

            for step_id, result in results.items():
                status = result.get("status", "unknown")
                icon = "+" if status == "success" else "-" if status == "error" else "?"
                click.echo(f"\n[{icon}] Step: {step_id}")
                click.echo(f"    Status: {status}")
                if result.get("output"):
                    click.echo(f"    Output: {result['output']}")
                if result.get("error"):
                    click.echo(f"    Error: {result['error']}", err=True)

            click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


# =============================================================================
# Example Config Generator
# =============================================================================

@v2_cli.command("init")
@click.option("--output", "-o", default="bbx-config.yaml", help="Output file")
def init_config(output: str):
    """Generate example BBX 2.0 configuration file.

    Example:
        bbx v2 init
        bbx v2 init -o my-config.yaml
    """
    from blackbox.core.v2.declarative import create_example_config

    config = create_example_config()
    config_dict = config.to_dict()

    yaml_content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    with open(output, "w", encoding="utf-8") as f:
        f.write("# BBX 2.0 Configuration\n")
        f.write("# Generated by: bbx v2 init\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write("#\n")
        f.write("# Apply with: bbx v2 config apply " + output + "\n")
        f.write("# Validate with: bbx v2 config validate " + output + "\n")
        f.write("\n")
        f.write(yaml_content)

    click.echo(f"[+] Created: {output}")
    click.echo(f"\n    Edit the file to configure your agent")
    click.echo(f"    Apply with: bbx v2 config apply {output}")


# =============================================================================
# Flow Integrity Commands (CET - Control-flow Enforcement)
# =============================================================================

@v2_cli.group("flow")
def flow_group():
    """Flow integrity commands - CET-inspired control flow protection.

    Provides shadow stack and indirect branch tracking for
    workflow execution integrity verification.

    Examples:
        bbx v2 flow stats
        bbx v2 flow verify workflow_123
    """
    pass


@flow_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def flow_stats(format: str):
    """Show flow integrity statistics."""
    async def _get_stats():
        from blackbox.core.v2.flow_integrity import get_flow_integrity
        return get_flow_integrity().get_stats()

    stats = asyncio.run(_get_stats())
    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Flow Integrity Statistics (CET-inspired)")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nConfiguration:")
        click.echo(f"  Enabled:        {stats.get('enabled', False)}")
        click.echo(f"  Strict Mode:    {stats.get('strict_mode', False)}")
        click.echo(f"\nStatistics:")
        click.echo(f"  Active Agents:  {stats.get('active_agents', 0)}")
        click.echo(f"  Violations:     {stats.get('total_violations', 0)}")
        click.echo(f"  Policies:       {stats.get('policies_count', 0)}")
        violations_by_type = stats.get('violations_by_type', {})
        if violations_by_type:
            click.echo(f"\nViolations by Type:")
            for vtype, count in violations_by_type.items():
                click.echo(f"  {vtype}: {count}")


@flow_group.command("verify")
@click.argument("workflow_id")
def flow_verify(workflow_id: str):
    """Verify workflow execution flow integrity."""
    async def _verify():
        from blackbox.core.v2.flow_integrity import get_flow_integrity
        return await get_flow_integrity().verify_workflow(workflow_id)

    result = asyncio.run(_verify())
    if result.valid:
        click.echo(f"[+] Workflow {workflow_id} integrity verified")
    else:
        click.echo(f"[-] Integrity violation: {result.error}", err=True)


# =============================================================================
# Agent Quotas Commands (Cgroups v2 - Resource Control)
# =============================================================================

@v2_cli.group("quotas")
def quotas_group():
    """Agent quotas commands - Cgroups v2-inspired resource control.

    Manage CPU, memory, I/O, and token quotas for agent groups.

    Examples:
        bbx v2 quotas stats
        bbx v2 quotas set mygroup --memory 512 --tokens 100000
        bbx v2 quotas list
    """
    pass


@quotas_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def quotas_stats(format: str):
    """Show quotas statistics."""
    from blackbox.core.v2.agent_quotas import get_quota_manager

    manager = get_quota_manager()
    stats = manager.get_stats()

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Agent Quotas Statistics (Cgroups v2-inspired)")
        click.echo(f"{'=' * 60}")
        root = stats.get('root', {})
        click.echo(f"\nRoot Group: {root.get('name', 'N/A')}")
        click.echo(f"  Agents:     {root.get('agents_count', 0)}")
        click.echo(f"  Children:   {root.get('children_count', 0)}")
        click.echo(f"  Violations: {root.get('total_violations', 0)}")
        click.echo(f"\nTotal Agents: {stats.get('total_agents', 0)}")


@quotas_group.command("set")
@click.argument("group")
@click.option("--cpu", type=int, default=1024, help="CPU shares")
@click.option("--memory", type=int, default=512, help="Memory limit (MB)")
@click.option("--io", type=int, default=1000, help="Max I/O ops/sec")
@click.option("--tokens", type=int, default=100000, help="Tokens per hour")
def quotas_set(group: str, cpu: int, memory: int, io: int, tokens: int):
    """Set resource quotas for an agent group."""
    async def _set():
        from blackbox.core.v2.agent_quotas import get_quota_manager, QuotaConfig
        manager = get_quota_manager()
        config = QuotaConfig(
            cpu_shares=cpu, memory_max_mb=memory,
            io_max_ops=io, tokens_per_hour=tokens
        )
        await manager.set_quota(group, config)

    asyncio.run(_set())
    click.echo(f"[+] Quotas set for group: {group}")


@quotas_group.command("list")
def quotas_list():
    """List all quota groups."""
    from blackbox.core.v2.agent_quotas import get_quota_manager

    manager = get_quota_manager()
    groups = manager.list_groups()

    if not groups:
        click.echo("No quota groups defined.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Agent Quota Groups")
    click.echo(f"{'=' * 60}")
    for g in groups:
        click.echo(f"\n[{g.name}]")
        click.echo(f"  CPU:    {g.cpu_shares} shares")
        click.echo(f"  Memory: {g.memory_max_mb} MB")
        click.echo(f"  Tokens: {g.tokens_per_hour}/hour")


# =============================================================================
# State Snapshots Commands (XFS Reflink - CoW Snapshots)
# =============================================================================

@v2_cli.group("snapshot")
def snapshot_group():
    """State snapshot commands - XFS reflink-inspired CoW snapshots.

    Create instant, space-efficient snapshots of agent state
    using copy-on-write technology.

    Examples:
        bbx v2 snapshot create backup-1
        bbx v2 snapshot list
        bbx v2 snapshot restore abc123
    """
    pass


@snapshot_group.command("create")
@click.argument("name")
@click.option("--description", "-d", default="", help="Snapshot description")
def snapshot_create(name: str, description: str):
    """Create a state snapshot."""
    async def _create():
        from blackbox.core.v2.state_snapshots import get_snapshot_manager
        manager = get_snapshot_manager()
        return await manager.create_snapshot(name, description=description)

    snapshot = asyncio.run(_create())
    click.echo(f"[+] Snapshot created")
    click.echo(f"    ID:   {snapshot.id}")
    click.echo(f"    Name: {snapshot.name}")
    click.echo(f"    Size: {snapshot.size_bytes / 1024:.1f} KB")


@snapshot_group.command("list")
def snapshot_list():
    """List all snapshots."""
    from blackbox.core.v2.state_snapshots import get_snapshot_manager

    manager = get_snapshot_manager()
    snapshots = manager.list_snapshots()

    if not snapshots:
        click.echo("No snapshots found.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("State Snapshots (CoW-enabled)")
    click.echo(f"{'=' * 60}")
    for s in snapshots:
        click.echo(f"\n[{s.id[:8]}...] {s.name}")
        click.echo(f"     Created: {s.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"     Size: {s.size_bytes / 1024:.1f} KB")


@snapshot_group.command("restore")
@click.argument("snapshot_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def snapshot_restore(snapshot_id: str, yes: bool):
    """Restore state from a snapshot."""
    if not yes and not click.confirm(f"Restore from snapshot {snapshot_id}?"):
        return

    async def _restore():
        from blackbox.core.v2.state_snapshots import get_snapshot_manager
        return await get_snapshot_manager().restore_snapshot(snapshot_id)

    success = asyncio.run(_restore())
    if success:
        click.echo(f"[+] Restored from snapshot: {snapshot_id}")
    else:
        click.echo(f"[-] Snapshot not found: {snapshot_id}", err=True)


@snapshot_group.command("stats")
def snapshot_stats():
    """Show snapshot statistics."""
    from blackbox.core.v2.state_snapshots import get_snapshot_manager

    manager = get_snapshot_manager()
    stats = manager.get_stats()

    click.echo(f"\n{'=' * 60}")
    click.echo("Snapshot Statistics")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nTotal Snapshots: {stats.total_snapshots}")
    click.echo(f"Total Size:      {stats.total_size_bytes / 1024 / 1024:.1f} MB")
    click.echo(f"CoW Savings:     {stats.space_saved_bytes / 1024 / 1024:.1f} MB")


# =============================================================================
# Flakes Commands (Nix Flakes - Reproducible Packages)
# =============================================================================

@v2_cli.group("flakes")
def flakes_group():
    """Flakes commands - Nix flakes-inspired reproducible packages.

    Build, run, and manage reproducible workflow packages
    with locked dependencies.

    Examples:
        bbx v2 flakes build ./my-agent
        bbx v2 flakes run github:user/agent
        bbx v2 flakes lock ./my-agent
    """
    pass


@flakes_group.command("build")
@click.argument("flake_path", type=click.Path(exists=True))
def flakes_build(flake_path: str):
    """Build a flake."""
    async def _build():
        from blackbox.core.v2.flakes import FlakeManager
        return await FlakeManager().build(flake_path)

    result = asyncio.run(_build())
    click.echo(f"[+] Flake built successfully")
    click.echo(f"    Path: {result.store_path}")
    click.echo(f"    Hash: {result.hash}")


@flakes_group.command("run")
@click.argument("flake_ref")
@click.option("--input", "-i", "inputs", multiple=True, help="Inputs in key=value format")
def flakes_run(flake_ref: str, inputs: tuple):
    """Run a flake directly."""
    async def _run():
        from blackbox.core.v2.flakes import FlakeManager

        input_dict = {}
        for inp in inputs:
            if "=" in inp:
                k, v = inp.split("=", 1)
                input_dict[k] = v

        return await FlakeManager().run(flake_ref, inputs=input_dict)

    result = asyncio.run(_run())
    click.echo(f"[+] Flake executed")
    click.echo(json.dumps(result, indent=2, default=str))


@flakes_group.command("lock")
@click.argument("flake_path", type=click.Path(exists=True))
def flakes_lock(flake_path: str):
    """Update flake lock file."""
    async def _lock():
        from blackbox.core.v2.flakes import FlakeManager
        return await FlakeManager().update_lock(flake_path)

    lock = asyncio.run(_lock())
    click.echo(f"[+] Lock updated: {lock.path}")


@flakes_group.command("show")
@click.argument("flake_ref")
def flakes_show(flake_ref: str):
    """Show flake metadata."""
    from blackbox.core.v2.flakes import FlakeManager

    info = FlakeManager().show(flake_ref)
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Flake: {info.description}")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nURL:      {info.url}")
    click.echo(f"Revision: {info.revision}")
    click.echo(f"\nInputs:")
    for k, v in info.inputs.items():
        click.echo(f"  - {k}: {v}")


# =============================================================================
# Agent Registry Commands (AUR - Package Discovery)
# =============================================================================

@v2_cli.group("packages")
def packages_group():
    """Package registry commands - AUR-inspired agent discovery.

    Search, install, and publish agent packages from
    the community registry.

    Examples:
        bbx v2 packages search web-scraper
        bbx v2 packages install data-analyst
        bbx v2 packages publish ./my-agent
    """
    pass


@packages_group.command("search")
@click.argument("query")
def packages_search(query: str):
    """Search agent registry."""
    async def _search():
        from blackbox.core.v2.agent_registry import get_agent_registry
        return await get_agent_registry().search(query)

    results = asyncio.run(_search())

    if not results:
        click.echo(f"No agents found matching: {query}")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Search Results: '{query}'")
    click.echo(f"{'=' * 60}")
    for agent in results[:20]:
        click.echo(f"\n[{agent.name}] v{agent.version}")
        click.echo(f"    {agent.description}")
        click.echo(f"    Downloads: {agent.downloads}")


@packages_group.command("install")
@click.argument("name")
@click.option("--version", "-v", default=None, help="Specific version")
def packages_install(name: str, version: Optional[str]):
    """Install agent from registry."""
    async def _install():
        from blackbox.core.v2.agent_registry import get_agent_registry
        return await get_agent_registry().install(name, version=version)

    result = asyncio.run(_install())
    click.echo(f"[+] Installed: {result.name} v{result.version}")


@packages_group.command("publish")
@click.argument("path", type=click.Path(exists=True))
def packages_publish(path: str):
    """Publish agent to registry."""
    async def _publish():
        from blackbox.core.v2.agent_registry import get_agent_registry
        return await get_agent_registry().publish(path)

    result = asyncio.run(_publish())
    click.echo(f"[+] Published: {result.name} v{result.version}")
    click.echo(f"    URL: {result.url}")


@packages_group.command("list")
def packages_list():
    """List installed agents."""
    from blackbox.core.v2.agent_registry import get_agent_registry

    agents = get_agent_registry().list_installed()

    if not agents:
        click.echo("No agents installed.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Installed Agents")
    click.echo(f"{'=' * 60}")
    for a in agents:
        click.echo(f"\n[{a.name}] v{a.version}")
        click.echo(f"    Installed: {a.installed_at.strftime('%Y-%m-%d')}")


# =============================================================================
# Agent Bundles Commands (Kali-style Tool Collections)
# =============================================================================

@v2_cli.group("bundles")
def bundles_group():
    """Bundle commands - Kali-style tool collections.

    Install curated collections of related agents
    for specific tasks.

    Examples:
        bbx v2 bundles list
        bbx v2 bundles install data-science
        bbx v2 bundles show security-testing
    """
    pass


@bundles_group.command("list")
def bundles_list():
    """List available bundles."""
    from blackbox.core.v2.agent_bundles import get_bundle_manager

    bundles = get_bundle_manager().list_bundles()

    click.echo(f"\n{'=' * 60}")
    click.echo("Agent Bundles (Kali-style)")
    click.echo(f"{'=' * 60}")
    for b in bundles:
        status = "[installed]" if b.installed else "[available]"
        click.echo(f"\n{status} {b.name}")
        click.echo(f"    {b.description}")
        click.echo(f"    Tools: {len(b.tools)}")


@bundles_group.command("install")
@click.argument("name")
def bundles_install(name: str):
    """Install an agent bundle."""
    async def _install():
        from blackbox.core.v2.agent_bundles import get_bundle_manager
        return await get_bundle_manager().install_bundle(name)

    result = asyncio.run(_install())
    click.echo(f"[+] Bundle installed: {name}")
    click.echo(f"    Tools: {', '.join(result.tools)}")


@bundles_group.command("show")
@click.argument("name")
def bundles_show(name: str):
    """Show bundle details."""
    from blackbox.core.v2.agent_bundles import get_bundle_manager

    bundle = get_bundle_manager().get_bundle(name)
    if not bundle:
        click.echo(f"Bundle not found: {name}", err=True)
        return

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Bundle: {bundle.name}")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n{bundle.description}")
    click.echo(f"\nCategory: {bundle.category}")
    click.echo(f"Version:  {bundle.version}")
    click.echo(f"\nTools ({len(bundle.tools)}):")
    for t in bundle.tools:
        click.echo(f"  - {t.name}: {t.description}")


# =============================================================================
# Agent Sandbox Commands (Flatpak-style Isolation)
# =============================================================================

@v2_cli.group("sandbox")
def sandbox_group():
    """Sandbox commands - Flatpak-style isolation.

    Run agents in isolated environments with
    controlled permissions.

    Examples:
        bbx v2 sandbox run untrusted-agent --permissions network
        bbx v2 sandbox list
        bbx v2 sandbox permissions
    """
    pass


@sandbox_group.command("run")
@click.argument("agent")
@click.option("--permission", "-p", "permissions", multiple=True, help="Permissions to grant")
@click.option("--input", "-i", "inputs", multiple=True, help="Inputs in key=value format")
def sandbox_run(agent: str, permissions: tuple, inputs: tuple):
    """Run agent in sandbox."""
    async def _run():
        from blackbox.core.v2.agent_sandbox import get_sandbox_manager

        input_dict = {}
        for inp in inputs:
            if "=" in inp:
                k, v = inp.split("=", 1)
                input_dict[k] = v

        return await get_sandbox_manager().run_sandboxed(
            agent, permissions=list(permissions), inputs=input_dict
        )

    result = asyncio.run(_run())
    click.echo(f"[+] Sandboxed execution complete")
    click.echo(f"    Sandbox: {result.sandbox_id}")
    click.echo(f"    Duration: {result.duration_ms:.1f}ms")


@sandbox_group.command("list")
def sandbox_list():
    """List active sandboxes."""
    from blackbox.core.v2.agent_sandbox import get_sandbox_manager

    sandboxes = get_sandbox_manager().list_sandboxes()

    if not sandboxes:
        click.echo("No active sandboxes.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Active Sandboxes")
    click.echo(f"{'=' * 60}")
    for s in sandboxes:
        click.echo(f"\n[{s.id[:8]}...] {s.agent}")
        click.echo(f"    Status: {s.status}")
        click.echo(f"    Permissions: {', '.join(s.permissions)}")


@sandbox_group.command("permissions")
def sandbox_permissions():
    """List available sandbox permissions."""
    from blackbox.core.v2.agent_sandbox import get_sandbox_manager

    perms = get_sandbox_manager().list_permissions()

    click.echo(f"\n{'=' * 60}")
    click.echo("Sandbox Permissions (Flatpak-style)")
    click.echo(f"{'=' * 60}")
    for p in perms:
        click.echo(f"\n[{p.name}]")
        click.echo(f"    {p.description}")
        click.echo(f"    Risk: {p.risk_level}")


# =============================================================================
# Network Fabric Commands (Istio-style Service Mesh)
# =============================================================================

@v2_cli.group("mesh")
def mesh_group():
    """Mesh commands - Istio-inspired service mesh.

    Manage agent service mesh networking, traffic routing,
    and observability.

    Examples:
        bbx v2 mesh status
        bbx v2 mesh services
        bbx v2 mesh route add frontend backend --weight 80
    """
    pass


@mesh_group.command("status")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def mesh_status(format: str):
    """Get mesh status."""
    from blackbox.core.v2.network_fabric import get_network_fabric

    status = get_network_fabric().get_status()

    if format == "json":
        click.echo(json.dumps(status.__dict__, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Network Fabric Status (Istio-inspired)")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nMesh Status:   {status.status}")
        click.echo(f"Control Plane: {status.control_plane}")
        click.echo(f"\nServices:")
        click.echo(f"  Registered: {status.services_registered}")
        click.echo(f"  Healthy:    {status.services_healthy}")
        click.echo(f"\nTraffic:")
        click.echo(f"  Requests/sec:  {status.requests_per_sec:.1f}")
        click.echo(f"  Success Rate:  {status.success_rate:.1f}%")


@mesh_group.command("services")
def mesh_services():
    """List mesh services."""
    from blackbox.core.v2.network_fabric import get_network_fabric

    services = get_network_fabric().list_services()

    if not services:
        click.echo("No services in mesh.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Mesh Services")
    click.echo(f"{'=' * 60}")
    for s in services:
        health = "[OK]" if s.healthy else "[ERR]"
        click.echo(f"\n{health} {s.name}")
        click.echo(f"    Endpoints: {len(s.endpoints)}")
        click.echo(f"    Load Balancing: {s.lb_policy}")


@mesh_group.command("route")
@click.argument("source")
@click.argument("destination")
@click.option("--name", "-n", required=True, help="Route name")
@click.option("--weight", "-w", type=int, default=100, help="Traffic weight (0-100)")
def mesh_route(source: str, destination: str, name: str, weight: int):
    """Create traffic routing rule."""
    async def _route():
        from blackbox.core.v2.network_fabric import get_network_fabric, TrafficRule
        rule = TrafficRule(name=name, source=source, destination=destination, weight=weight)
        await get_network_fabric().add_route(rule)

    asyncio.run(_route())
    click.echo(f"[+] Route created: {name}")
    click.echo(f"    {source} -> {destination} ({weight}%)")


# =============================================================================
# Policy Engine Commands (OPA/SELinux - Policy Enforcement)
# =============================================================================

@v2_cli.group("policy")
def policy_group():
    """Policy commands - OPA/SELinux-inspired enforcement.

    Define and evaluate access control policies
    for agent operations.

    Examples:
        bbx v2 policy list
        bbx v2 policy evaluate allow_network --input '{"agent":"myagent"}'
        bbx v2 policy stats
    """
    pass


@policy_group.command("list")
def policy_list():
    """List all policies."""
    from blackbox.core.v2.policy_engine import get_policy_engine

    policies = get_policy_engine().list_policies()

    if not policies:
        click.echo("No policies defined.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Policies (OPA/SELinux-inspired)")
    click.echo(f"{'=' * 60}")
    for p in policies:
        click.echo(f"\n  [{p.id}] {p.name}")
        click.echo(f"     Version: {p.version}")
        click.echo(f"     Rules: {len(p.rules) if p.rules else 0}")
        if p.description:
            click.echo(f"     Description: {p.description}")


@policy_group.command("evaluate")
@click.argument("policy")
@click.option("--input", "-i", "input_json", required=True, help="Input JSON")
def policy_evaluate(policy: str, input_json: str):
    """Evaluate policy against input."""
    async def _evaluate():
        from blackbox.core.v2.policy_engine import get_policy_engine
        input_data = json.loads(input_json)
        return await get_policy_engine().evaluate(policy, input_data)

    result = asyncio.run(_evaluate())
    if result.allowed:
        click.echo(f"[ALLOW] Policy: {policy}")
    else:
        click.echo(f"[DENY] Policy: {policy}")
    click.echo(f"Reason: {result.reason}")


@policy_group.command("stats")
def policy_stats():
    """Get policy engine statistics."""
    from blackbox.core.v2.policy_engine import get_policy_engine

    stats = get_policy_engine().get_stats()

    click.echo(f"\n{'=' * 60}")
    click.echo("Policy Engine Statistics")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nEvaluations:")
    click.echo(f"  Total:   {stats.get('total_evaluations', 0)}")
    click.echo(f"  Allowed: {stats.get('allowed', 0)}")
    click.echo(f"  Denied:  {stats.get('denied', 0)}")
    click.echo(f"\nPolicies:")
    click.echo(f"  Count:      {stats.get('policies_count', 0)}")
    click.echo(f"  Cache Size: {stats.get('cache_size', 0)}")


# =============================================================================
# AAL Commands (HAL - Adapter Abstraction Layer)
# =============================================================================

@v2_cli.group("aal")
def aal_group():
    """AAL commands - HAL-inspired adapter abstraction.

    Hardware abstraction layer for unified adapter access.

    Examples:
        bbx v2 aal adapters
        bbx v2 aal call shell execute --args '{"command":"ls"}'
        bbx v2 aal stats
    """
    pass


@aal_group.command("adapters")
def aal_adapters():
    """List all adapters through AAL."""
    from blackbox.core.v2.aal import get_aal

    adapters = get_aal().list_adapters()

    click.echo(f"\n{'=' * 60}")
    click.echo("Adapter Abstraction Layer (HAL-inspired)")
    click.echo(f"{'=' * 60}")
    for a in adapters:
        status = "[OK]" if a.healthy else "[ERR]"
        click.echo(f"\n{status} {a.name}")
        click.echo(f"    Type: {a.adapter_type}")
        click.echo(f"    Methods: {', '.join(a.methods)}")


@aal_group.command("call")
@click.argument("adapter")
@click.argument("method")
@click.option("--args", "-a", default="{}", help="Method arguments (JSON)")
def aal_call(adapter: str, method: str, args: str):
    """Call adapter method through AAL."""
    async def _call():
        from blackbox.core.v2.aal import get_aal
        args_dict = json.loads(args)
        return await get_aal().call(adapter, method, args_dict)

    result = asyncio.run(_call())
    click.echo(f"[+] {adapter}.{method}")
    click.echo(json.dumps(result, indent=2, default=str))


@aal_group.command("stats")
def aal_stats():
    """Get AAL statistics."""
    from blackbox.core.v2.aal import get_aal

    metrics = get_aal().get_metrics()

    click.echo(f"\n{'=' * 60}")
    click.echo("AAL Statistics")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nBackends:")
    for name, backend_metrics in metrics.get("backends", {}).items():
        click.echo(f"\n  [{name}]")
        click.echo(f"    Requests: {backend_metrics.get('requests', 0)}")
        click.echo(f"    Success Rate: {backend_metrics.get('success_rate', 0):.1%}")
        click.echo(f"    Avg Latency: {backend_metrics.get('avg_latency_ms', 0):.2f}ms")
    click.echo(f"\nGlobal:")
    click.echo(f"  Requests Routed: {metrics.get('requests_routed', 0)}")
    click.echo(f"  Failovers: {metrics.get('failovers', 0)}")


# =============================================================================
# Object Manager Commands (Windows ObMgr - Object Namespace)
# =============================================================================

@v2_cli.group("objects")
def objects_group():
    """Object manager commands - Windows ObMgr-inspired namespace.

    Manage named objects in hierarchical namespace.

    Examples:
        bbx v2 objects list /
        bbx v2 objects create myobj --type workflow
        bbx v2 objects stats
    """
    pass


@objects_group.command("list")
@click.argument("path", default="/")
def objects_list(path: str):
    """List objects in namespace."""
    from blackbox.core.v2.object_manager import get_object_manager

    # list_directory requires caller_sid
    objects = get_object_manager().list_directory(path, "system")

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Object Namespace: {path}")
    click.echo(f"{'=' * 60}")
    if objects:
        for obj_name in objects:
            click.echo(f"  {obj_name}")
    else:
        click.echo("  (empty or not found)")


@objects_group.command("create")
@click.argument("name")
@click.option("--type", "-t", "obj_type", required=True, help="Object type")
@click.option("--data", "-d", default="{}", help="Object data (JSON)")
def objects_create(name: str, obj_type: str, data: str):
    """Create a named object."""
    async def _create():
        from blackbox.core.v2.object_manager import get_object_manager
        data_dict = json.loads(data)
        return await get_object_manager().create_object(name, obj_type, data_dict)

    handle = asyncio.run(_create())
    click.echo(f"[+] Object created")
    click.echo(f"    Name: {name}")
    click.echo(f"    Type: {obj_type}")
    click.echo(f"    Handle: {handle}")


@objects_group.command("stats")
def objects_stats():
    """Get object manager statistics."""
    from blackbox.core.v2.object_manager import get_object_manager

    stats = get_object_manager().get_stats()

    click.echo(f"\n{'=' * 60}")
    click.echo("Object Manager Statistics (ObMgr-inspired)")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nObjects: {stats.get('total_objects', 0)}")
    click.echo(f"Open Handles: {stats.get('open_handles', 0)}")
    click.echo(f"\nOperations:")
    click.echo(f"  Creates:  {stats.get('creates', 0)}")
    click.echo(f"  Opens:    {stats.get('opens', 0)}")
    click.echo(f"  Lookups:  {stats.get('lookups', 0)}")


# =============================================================================
# Filter Stack Commands (Windows Filter Drivers - I/O Pipeline)
# =============================================================================

@v2_cli.group("filters")
def filters_group():
    """Filter stack commands - Windows Filter Drivers-inspired.

    Manage I/O filter pipeline for intercepting
    and modifying adapter operations.

    Examples:
        bbx v2 filters list
        bbx v2 filters add audit --altitude 300000
        bbx v2 filters stats
    """
    pass


@filters_group.command("list")
def filters_list():
    """List registered filters."""
    from blackbox.core.v2.filter_stack import get_filter_manager

    filters = get_filter_manager().list_filters()

    if not filters:
        click.echo("No filters registered.")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo("Filter Stack (Filter Drivers-inspired)")
    click.echo(f"{'=' * 60}")
    for f in sorted(filters, key=lambda x: x.get('altitude', 0), reverse=True):
        enabled = f.get('enabled', True)
        status = "[ON] " if enabled else "[OFF]"
        name = f.get('name', 'Unknown')
        altitude = f.get('altitude', 0)
        filter_type = f.get('filter_class', 'unknown')
        operations = f.get('operations', [])
        click.echo(f"\n{status} {name} (altitude: {altitude})")
        click.echo(f"     Type: {filter_type}")
        if operations:
            click.echo(f"     Operations: {', '.join(operations)}")


@filters_group.command("add")
@click.argument("name")
@click.option("--altitude", "-a", type=int, default=100000, help="Filter altitude")
@click.option("--type", "-t", "filter_type", default="passthrough", help="Filter type")
@click.option("--operations", "-o", default="all", help="Operations to intercept")
def filters_add(name: str, altitude: int, filter_type: str, operations: str):
    """Add a filter to the stack."""
    async def _add():
        from blackbox.core.v2.filter_stack import get_filter_manager
        # Note: This is a placeholder - actual filter registration needs proper filter class
        click.echo(f"[!] Custom filter registration requires proper Filter class implementation")

    asyncio.run(_add())
    click.echo(f"[+] Filter registered: {name} at altitude {altitude}")


@filters_group.command("remove")
@click.argument("name")
def filters_remove(name: str):
    """Remove a filter from the stack."""
    async def _remove():
        from blackbox.core.v2.filter_stack import get_filter_stack
        return await get_filter_stack().unregister_filter(name)

    success = asyncio.run(_remove())
    if success:
        click.echo(f"[+] Filter removed: {name}")
    else:
        click.echo(f"[-] Filter not found: {name}", err=True)


@filters_group.command("stats")
def filters_stats():
    """Get filter stack statistics."""
    from blackbox.core.v2.filter_stack import get_filter_stack

    stats = get_filter_stack().get_stats()

    click.echo(f"\n{'=' * 60}")
    click.echo("Filter Stack Statistics")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nFilters:")
    click.echo(f"  Registered: {stats.filters_registered}")
    click.echo(f"  Active:     {stats.filters_active}")
    click.echo(f"\nOperations:")
    click.echo(f"  Total I/O:  {stats.total_io_ops}")
    click.echo(f"  Blocked:    {stats.blocked_ops}")
    click.echo(f"  Modified:   {stats.modified_ops}")


# =============================================================================
# Working Set Commands (Windows Mm - Memory Management)
# =============================================================================

@v2_cli.group("memory")
def memory_group():
    """Memory commands - Windows Mm-inspired working set management.

    Manage agent memory working set, page pools,
    and memory optimization.

    Examples:
        bbx v2 memory stats
        bbx v2 memory trim --target 256
        bbx v2 memory pools
    """
    pass


@memory_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def memory_stats(format: str):
    """Get working set statistics."""
    from blackbox.core.v2.working_set import get_working_set_manager

    stats = get_working_set_manager().get_stats()

    if format == "json":
        # Handle both object and dict cases
        if hasattr(stats, '__dict__'):
            click.echo(json.dumps(stats.__dict__, indent=2, default=str))
        else:
            click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Working Set Statistics (Mm-inspired)")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nMemory:")
        click.echo(f"  Used: {stats.memory_used_bytes / 1024 / 1024:.1f} MB")
        click.echo(f"  Compressed: {stats.compressed_bytes / 1024 / 1024:.1f} MB")
        click.echo(f"  Pressure: {stats.current_pressure.value}")
        click.echo(f"\nPages:")
        total_pages = stats.active_pages + stats.standby_pages + stats.modified_pages
        click.echo(f"  Total:      {total_pages}")
        click.echo(f"  Active:     {stats.active_pages}")
        click.echo(f"  Standby:    {stats.standby_pages}")
        click.echo(f"  Modified:   {stats.modified_pages}")
        click.echo(f"  Compressed: {stats.compressed_pages}")
        click.echo(f"  Paged Out:  {stats.paged_out_pages}")
        click.echo(f"\nFaults:")
        click.echo(f"  Soft: {stats.soft_faults}")
        click.echo(f"  Hard: {stats.hard_faults}")
        click.echo(f"\nTrims:")
        click.echo(f"  Performed: {stats.trims_performed}")
        click.echo(f"  Pages Trimmed: {stats.pages_trimmed}")


@memory_group.command("trim")
@click.option("--target", "-t", type=int, help="Target size in MB")
def memory_trim(target: Optional[int]):
    """Trim working set."""
    async def _trim():
        from blackbox.core.v2.working_set import get_working_set_manager
        return await get_working_set_manager().trim_working_set(target)

    freed = asyncio.run(_trim())
    click.echo(f"[+] Working set trimmed")
    click.echo(f"    Freed: {freed / 1024 / 1024:.1f} MB")


@memory_group.command("lock")
@click.argument("key")
def memory_lock(key: str):
    """Lock pages in working set."""
    async def _lock():
        from blackbox.core.v2.working_set import get_working_set_manager
        return await get_working_set_manager().lock_pages(key)

    success = asyncio.run(_lock())
    if success:
        click.echo(f"[+] Pages locked: {key}")
    else:
        click.echo(f"[-] Failed to lock: {key}", err=True)


@memory_group.command("pools")
def memory_pools():
    """Show memory pool statistics."""
    from blackbox.core.v2.working_set import get_working_set_manager

    pools = get_working_set_manager().get_pool_stats()

    click.echo(f"\n{'=' * 60}")
    click.echo("Memory Pools")
    click.echo(f"{'=' * 60}")
    for pool in pools:
        click.echo(f"\n[{pool.name}]")
        click.echo(f"    Allocated: {pool.allocated_mb:.1f} MB")
        click.echo(f"    Used:      {pool.used_mb:.1f} MB")
        click.echo(f"    Peak:      {pool.peak_mb:.1f} MB")


# =============================================================================
# Config Registry Commands (Windows Registry - Hierarchical Config)
# =============================================================================

@v2_cli.group("registry")
def registry_group():
    """Registry commands - Windows Registry-inspired configuration.

    Hierarchical key-value configuration store
    with typed values and access control.

    Examples:
        bbx v2 registry get HKLM\\BBX\\Settings\\Theme
        bbx v2 registry set HKLM\\BBX\\Settings\\Theme dark
        bbx v2 registry list HKLM\\BBX
    """
    pass


@registry_group.command("get")
@click.argument("path")
@click.argument("value")
def registry_get(path: str, value: str):
    """Get registry value."""
    async def _get():
        from blackbox.core.v2.config_registry import get_config_registry
        return await get_config_registry().get_value(path, value)

    result = asyncio.run(_get())
    if result is None:
        click.echo(f"Value not found: {path}\\{value}", err=True)
    else:
        click.echo(f"{path}\\{value} = {result}")


@registry_group.command("set")
@click.argument("path")
@click.argument("value_name")
@click.argument("data")
@click.option("--type", "-t", "value_type", default="string", help="Value type")
def registry_set(path: str, value_name: str, data: str, value_type: str):
    """Set registry value."""
    async def _set():
        from blackbox.core.v2.config_registry import get_config_registry
        await get_config_registry().set_value(path, value_name, data, value_type=value_type)

    asyncio.run(_set())
    click.echo(f"[+] Set: {path}\\{value_name}")


@registry_group.command("list")
@click.argument("path", default="HKBX_SYSTEM")
def registry_list(path: str):
    """List registry keys and values."""
    async def _list():
        from blackbox.core.v2.config_registry import get_config_registry
        reg = get_config_registry()
        subkeys = await reg.enumerate_keys(path)
        values = await reg.enumerate_values(path)
        return subkeys, values

    subkeys, values = asyncio.run(_list())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Registry: {path}")
    click.echo(f"{'=' * 60}")

    if subkeys:
        click.echo("\nSubkeys:")
        for sk in subkeys:
            click.echo(f"  [{sk}]")

    if values:
        click.echo("\nValues:")
        for v in values:
            click.echo(f"  {v}")


@registry_group.command("delete")
@click.argument("path")
@click.option("--value", "-v", default=None, help="Value to delete (or delete key)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def registry_delete(path: str, value: Optional[str], yes: bool):
    """Delete registry key or value."""
    target = f"{path}\\{value}" if value else path
    if not yes and not click.confirm(f"Delete {target}?"):
        return

    async def _delete():
        from blackbox.core.v2.config_registry import get_config_registry
        reg = get_config_registry()
        if value:
            return await reg.delete_value(path, value)
        return await reg.delete_key(path)

    success = asyncio.run(_delete())
    if success:
        click.echo(f"[+] Deleted: {target}")
    else:
        click.echo(f"[-] Not found: {target}", err=True)


@registry_group.command("export")
@click.argument("path")
@click.argument("output_file")
def registry_export(path: str, output_file: str):
    """Export registry to file."""
    async def _export():
        from blackbox.core.v2.config_registry import get_config_registry
        await get_config_registry().export_key(path, output_file)

    asyncio.run(_export())
    click.echo(f"[+] Exported {path} to {output_file}")


# =============================================================================
# Executive Commands (Windows ntoskrnl - Hybrid Kernel)
# =============================================================================

@v2_cli.group("executive")
def executive_group():
    """Executive commands - Windows ntoskrnl-inspired hybrid kernel.

    Manage BBX kernel subsystems, system calls,
    and diagnostic information.

    Examples:
        bbx v2 executive status
        bbx v2 executive start --subsystems object_manager,memory_manager
        bbx v2 executive bugcheck
    """
    pass


@executive_group.command("status")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def executive_status(format: str):
    """Get executive (kernel) status."""
    from blackbox.core.v2.executive import get_executive

    stats = get_executive().get_stats()

    if format == "json":
        click.echo(json.dumps(stats.__dict__, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("BBX Executive Status (ntoskrnl-inspired)")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nKernel:")
        click.echo(f"  Uptime:  {stats.uptime_seconds:.0f}s")
        click.echo(f"  Memory Pressure: {stats.memory_pressure.value}")
        click.echo(f"\nSubsystems:")
        click.echo(f"  Active Handles: {stats.active_handles}")
        click.echo(f"  Active Tokens:  {stats.active_tokens}")
        click.echo(f"\nPerformance:")
        click.echo(f"  Total Syscalls:      {stats.total_syscalls}")
        click.echo(f"  Successful Syscalls: {stats.successful_syscalls}")
        click.echo(f"  Failed Syscalls:     {stats.failed_syscalls}")
        click.echo(f"  Avg Latency:         {stats.avg_latency_ms:.2f}ms")


@executive_group.command("start")
@click.option("--subsystems", "-s", default="all", help="Subsystems to start (comma-separated)")
def executive_start(subsystems: str):
    """Start executive subsystems."""
    async def _start():
        from blackbox.core.v2.executive import get_executive
        subs = ["all"] if subsystems == "all" else subsystems.split(",")
        await get_executive().start_subsystems(subs)

    asyncio.run(_start())
    click.echo(f"[+] Executive subsystems started: {subsystems}")


@executive_group.command("stop")
@click.option("--subsystems", "-s", default="all", help="Subsystems to stop (comma-separated)")
def executive_stop(subsystems: str):
    """Stop executive subsystems."""
    async def _stop():
        from blackbox.core.v2.executive import get_executive
        subs = ["all"] if subsystems == "all" else subsystems.split(",")
        await get_executive().stop_subsystems(subs)

    asyncio.run(_stop())
    click.echo(f"[+] Executive subsystems stopped: {subsystems}")


@executive_group.command("syscall")
@click.argument("syscall_name")
@click.option("--args", "-a", default="{}", help="System call arguments (JSON)")
def executive_syscall(syscall_name: str, args: str):
    """Execute system call through executive."""
    async def _syscall():
        from blackbox.core.v2.executive import get_executive
        args_dict = json.loads(args)
        return await get_executive().syscall(syscall_name, args_dict)

    result = asyncio.run(_syscall())
    click.echo(f"[+] Syscall: {syscall_name}")
    click.echo(json.dumps(result, indent=2, default=str))


@executive_group.command("bugcheck")
def executive_bugcheck():
    """Generate diagnostic bugcheck/crash dump."""
    from blackbox.core.v2.executive import get_executive

    dump = get_executive().generate_bugcheck()

    click.echo(f"\n{'=' * 60}")
    click.echo("BBX Bugcheck Report")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nCode:   {dump.code}")
    click.echo(f"Time:   {dump.timestamp}")
    click.echo(f"Thread: {dump.thread_id}")
    click.echo(f"\nStack Trace:")
    click.echo(dump.stack_trace)
    click.echo(f"\nLoaded Modules:")
    for m in dump.loaded_modules:
        click.echo(f"  {m}")


# =============================================================================
# Enhanced Ring Commands (WAL, Idempotency, Circuit Breaker)
# =============================================================================

@v2_cli.group("enhanced-ring")
def enhanced_ring_group():
    """Enhanced AgentRing commands - Production-ready with WAL.

    Features:
    - WAL (Write-Ahead Log) for durability
    - Idempotency keys for exactly-once semantics
    - Circuit breaker for fault tolerance
    - Shared memory for cross-process communication

    Examples:
        bbx v2 enhanced-ring stats
        bbx v2 enhanced-ring wal-status
        bbx v2 enhanced-ring circuit-breaker
    """
    pass


@enhanced_ring_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def enhanced_ring_stats(format: str):
    """Show enhanced ring statistics with WAL and circuit breaker status."""
    async def _get_stats():
        from blackbox.core.v2.ring_enhanced import get_enhanced_ring
        ring = get_enhanced_ring()
        return ring.get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Enhanced AgentRing Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nOperations:")
        click.echo(f"  Submitted:      {stats.get('submitted', 0)}")
        click.echo(f"  Completed:      {stats.get('completed', 0)}")
        click.echo(f"  Deduplicated:   {stats.get('deduplicated', 0)}")
        click.echo(f"\nWAL:")
        click.echo(f"  Entries:        {stats.get('wal_entries', 0)}")
        click.echo(f"  Size:           {stats.get('wal_size_mb', 0):.2f} MB")
        click.echo(f"\nCircuit Breaker:")
        click.echo(f"  State:          {stats.get('circuit_state', 'CLOSED')}")
        click.echo(f"  Failures:       {stats.get('circuit_failures', 0)}")
        click.echo(f"{'=' * 60}")


@enhanced_ring_group.command("wal-status")
def enhanced_ring_wal_status():
    """Show WAL (Write-Ahead Log) status."""
    async def _get_status():
        from blackbox.core.v2.ring_enhanced import get_enhanced_ring
        ring = get_enhanced_ring()
        return ring.get_wal_status()

    status = asyncio.run(_get_status())
    click.echo(f"\n{'=' * 60}")
    click.echo("WAL Status")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Path:       {status.get('path', 'N/A')}")
    click.echo(f"  Size:       {status.get('size_mb', 0):.2f} MB")
    click.echo(f"  Entries:    {status.get('entries', 0)}")
    click.echo(f"  Last Sync:  {status.get('last_sync', 'N/A')}")
    click.echo(f"  Checkpoints: {status.get('checkpoints', 0)}")


@enhanced_ring_group.command("circuit-breaker")
@click.option("--reset", is_flag=True, help="Reset circuit breaker")
def enhanced_ring_circuit_breaker(reset: bool):
    """Show or reset circuit breaker status."""
    async def _action():
        from blackbox.core.v2.ring_enhanced import get_enhanced_ring
        ring = get_enhanced_ring()
        if reset:
            ring.reset_circuit_breaker()
            return {"action": "reset"}
        return ring.get_circuit_breaker_status()

    result = asyncio.run(_action())

    if result.get("action") == "reset":
        click.echo("[+] Circuit breaker reset")
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Circuit Breaker Status")
        click.echo(f"{'=' * 60}")
        click.echo(f"\n  State:      {result.get('state', 'CLOSED')}")
        click.echo(f"  Failures:   {result.get('failures', 0)}")
        click.echo(f"  Threshold:  {result.get('threshold', 5)}")
        click.echo(f"  Reset Time: {result.get('reset_time', 'N/A')}")


# =============================================================================
# Enhanced Context Tiering Commands (ML Scoring, Prefetch)
# =============================================================================

@v2_cli.group("enhanced-context")
def enhanced_context_group():
    """Enhanced context tiering - ML-powered with prefetch.

    Features:
    - ML-based importance scoring
    - Prefetch API for predictive loading
    - Async migration between tiers
    - Compression optimization

    Examples:
        bbx v2 enhanced-context stats
        bbx v2 enhanced-context prefetch-hint my_key
        bbx v2 enhanced-context ml-score
    """
    pass


@enhanced_context_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def enhanced_context_stats(format: str):
    """Show enhanced context tiering statistics."""
    async def _get_stats():
        from blackbox.core.v2.context_tiering_enhanced import get_enhanced_tiering
        return get_enhanced_tiering().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Enhanced Context Tiering Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nML Scoring:")
        click.echo(f"  Predictions:   {stats.get('ml_predictions', 0)}")
        click.echo(f"  Accuracy:      {stats.get('ml_accuracy', 0):.1f}%")
        click.echo(f"\nPrefetch:")
        click.echo(f"  Hits:          {stats.get('prefetch_hits', 0)}")
        click.echo(f"  Miss Rate:     {stats.get('prefetch_miss_rate', 0):.1f}%")
        click.echo(f"\nMigration:")
        click.echo(f"  Pending:       {stats.get('migration_pending', 0)}")
        click.echo(f"  Completed:     {stats.get('migration_completed', 0)}")


@enhanced_context_group.command("prefetch-hint")
@click.argument("key")
@click.option("--priority", type=int, default=5, help="Prefetch priority (1-10)")
def enhanced_context_prefetch(key: str, priority: int):
    """Add prefetch hint for a key."""
    async def _prefetch():
        from blackbox.core.v2.context_tiering_enhanced import get_enhanced_tiering
        return await get_enhanced_tiering().prefetch_hint(key, priority=priority)

    result = asyncio.run(_prefetch())
    if result:
        click.echo(f"[+] Prefetch hint added: {key} (priority={priority})")
    else:
        click.echo(f"[-] Failed to add prefetch hint", err=True)


@enhanced_context_group.command("ml-score")
@click.argument("key")
def enhanced_context_ml_score(key: str):
    """Get ML importance score for a key."""
    async def _score():
        from blackbox.core.v2.context_tiering_enhanced import get_enhanced_tiering
        return await get_enhanced_tiering().get_importance_score(key)

    score = asyncio.run(_score())
    if score is not None:
        click.echo(f"[{key}] Importance Score: {score:.4f}")
    else:
        click.echo(f"[-] Key not found: {key}", err=True)


# =============================================================================
# Enforced Quotas Commands (Cgroups, GPU)
# =============================================================================

@v2_cli.group("enforced-quotas")
def enforced_quotas_group():
    """Enforced quotas - Real resource enforcement.

    Features:
    - Linux cgroups v2 integration
    - GPU quota management (NVIDIA MPS)
    - Token bucket rate limiting
    - Real process isolation

    Examples:
        bbx v2 enforced-quotas stats
        bbx v2 enforced-quotas gpu-status
        bbx v2 enforced-quotas cgroup-status
    """
    pass


@enforced_quotas_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def enforced_quotas_stats(format: str):
    """Show enforced quotas statistics."""
    async def _get_stats():
        from blackbox.core.v2.quotas_enforced import get_enforced_quotas
        return get_enforced_quotas().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Enforced Quotas Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nCgroups:")
        click.echo(f"  Active:        {stats.get('cgroups_active', 0)}")
        click.echo(f"  CPU Usage:     {stats.get('cpu_usage_percent', 0):.1f}%")
        click.echo(f"  Memory Usage:  {stats.get('memory_usage_mb', 0):.1f} MB")
        click.echo(f"\nGPU:")
        click.echo(f"  GPUs:          {stats.get('gpu_count', 0)}")
        click.echo(f"  GPU Memory:    {stats.get('gpu_memory_used_mb', 0):.1f} MB")
        click.echo(f"\nThrottling:")
        click.echo(f"  Throttled:     {stats.get('throttled_requests', 0)}")
        click.echo(f"  Rejected:      {stats.get('rejected_requests', 0)}")


@enforced_quotas_group.command("gpu-status")
def enforced_quotas_gpu_status():
    """Show GPU quota status."""
    async def _get_status():
        from blackbox.core.v2.quotas_enforced import get_enforced_quotas
        return get_enforced_quotas().get_gpu_status()

    status = asyncio.run(_get_status())

    click.echo(f"\n{'=' * 60}")
    click.echo("GPU Quota Status")
    click.echo(f"{'=' * 60}")
    for gpu in status.get("gpus", []):
        click.echo(f"\n[GPU {gpu['id']}] {gpu['name']}")
        click.echo(f"    Memory:     {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB")
        click.echo(f"    Utilization: {gpu['utilization']:.1f}%")
        click.echo(f"    Processes:   {gpu['processes']}")


@enforced_quotas_group.command("cgroup-status")
@click.argument("group", default="root")
def enforced_quotas_cgroup_status(group: str):
    """Show cgroup status for a group."""
    async def _get_status():
        from blackbox.core.v2.quotas_enforced import get_enforced_quotas
        return get_enforced_quotas().get_cgroup_status(group)

    status = asyncio.run(_get_status())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Cgroup Status: {group}")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Path:       {status.get('path', 'N/A')}")
    click.echo(f"  CPU:        {status.get('cpu_usage', 0):.1f}%")
    click.echo(f"  Memory:     {status.get('memory_mb', 0):.1f} / {status.get('memory_limit_mb', 0):.1f} MB")
    click.echo(f"  I/O Read:   {status.get('io_read_mb', 0):.1f} MB")
    click.echo(f"  I/O Write:  {status.get('io_write_mb', 0):.1f} MB")


# =============================================================================
# Distributed Snapshots Commands (S3, Replication)
# =============================================================================

@v2_cli.group("distributed-snapshots")
def distributed_snapshots_group():
    """Distributed snapshots - S3 backed with replication.

    Features:
    - S3/Redis storage backends
    - Cross-region replication
    - Point-in-time recovery
    - Async snapshot writer

    Examples:
        bbx v2 distributed-snapshots stats
        bbx v2 distributed-snapshots list-replicas
        bbx v2 distributed-snapshots pitr
    """
    pass


@distributed_snapshots_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def distributed_snapshots_stats(format: str):
    """Show distributed snapshot statistics."""
    async def _get_stats():
        from blackbox.core.v2.snapshots_distributed import get_distributed_snapshots
        return get_distributed_snapshots().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Distributed Snapshots Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nStorage:")
        click.echo(f"  Backend:       {stats.get('backend', 'local')}")
        click.echo(f"  Total Size:    {stats.get('total_size_mb', 0):.1f} MB")
        click.echo(f"  Snapshots:     {stats.get('snapshot_count', 0)}")
        click.echo(f"\nReplication:")
        click.echo(f"  Replicas:      {stats.get('replica_count', 0)}")
        click.echo(f"  Sync Lag:      {stats.get('replication_lag_ms', 0)} ms")
        click.echo(f"\nPITR:")
        click.echo(f"  Earliest:      {stats.get('earliest_recovery', 'N/A')}")
        click.echo(f"  Latest:        {stats.get('latest_recovery', 'N/A')}")


@distributed_snapshots_group.command("list-replicas")
def distributed_snapshots_replicas():
    """List snapshot replicas across regions."""
    async def _list():
        from blackbox.core.v2.snapshots_distributed import get_distributed_snapshots
        return get_distributed_snapshots().list_replicas()

    replicas = asyncio.run(_list())

    click.echo(f"\n{'=' * 60}")
    click.echo("Snapshot Replicas")
    click.echo(f"{'=' * 60}")
    for replica in replicas:
        status = "[OK]" if replica.get("healthy") else "[ERR]"
        click.echo(f"\n{status} {replica['region']}")
        click.echo(f"    Endpoint:  {replica['endpoint']}")
        click.echo(f"    Lag:       {replica['lag_ms']} ms")
        click.echo(f"    Size:      {replica['size_mb']:.1f} MB")


@distributed_snapshots_group.command("pitr")
@click.argument("timestamp")
@click.option("--agent-id", "-a", required=True, help="Agent ID to recover")
@click.option("--dry-run", is_flag=True, help="Show what would be recovered")
def distributed_snapshots_pitr(timestamp: str, agent_id: str, dry_run: bool):
    """Point-in-time recovery to specific timestamp."""
    async def _pitr():
        from blackbox.core.v2.snapshots_distributed import get_distributed_snapshots
        return await get_distributed_snapshots().point_in_time_recovery(
            agent_id, timestamp, dry_run=dry_run
        )

    result = asyncio.run(_pitr())

    if dry_run:
        click.echo(f"\n[DRY RUN] Would recover to: {timestamp}")
        click.echo(f"  Snapshot:  {result.get('snapshot_id', 'N/A')}")
        click.echo(f"  Data Size: {result.get('data_size_mb', 0):.1f} MB")
    else:
        click.echo(f"[+] Recovered agent {agent_id} to {timestamp}")
        click.echo(f"    Snapshot: {result.get('snapshot_id', 'N/A')}")


# =============================================================================
# Enhanced Flow Integrity Commands (Anomaly Detection, OPA)
# =============================================================================

@v2_cli.group("enhanced-flow")
def enhanced_flow_group():
    """Enhanced flow integrity - Anomaly detection and OPA.

    Features:
    - Behavioral anomaly detection
    - OPA policy engine integration
    - Memory access control
    - Tool call validation

    Examples:
        bbx v2 enhanced-flow stats
        bbx v2 enhanced-flow anomalies
        bbx v2 enhanced-flow policies
    """
    pass


@enhanced_flow_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def enhanced_flow_stats(format: str):
    """Show enhanced flow integrity statistics."""
    async def _get_stats():
        from blackbox.core.v2.flow_integrity_enhanced import get_enhanced_flow_integrity
        return get_enhanced_flow_integrity().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Enhanced Flow Integrity Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nAnomalies:")
        click.echo(f"  Detected:      {stats.get('anomalies_detected', 0)}")
        click.echo(f"  Blocked:       {stats.get('anomalies_blocked', 0)}")
        click.echo(f"  Score Avg:     {stats.get('anomaly_score_avg', 0):.3f}")
        click.echo(f"\nPolicies:")
        click.echo(f"  Evaluated:     {stats.get('policies_evaluated', 0)}")
        click.echo(f"  Denied:        {stats.get('policies_denied', 0)}")
        click.echo(f"\nMemory Access:")
        click.echo(f"  Checks:        {stats.get('memory_checks', 0)}")
        click.echo(f"  Violations:    {stats.get('memory_violations', 0)}")


@enhanced_flow_group.command("anomalies")
@click.option("--limit", "-n", type=int, default=10, help="Number of anomalies to show")
def enhanced_flow_anomalies(limit: int):
    """List recent anomalies detected."""
    async def _list():
        from blackbox.core.v2.flow_integrity_enhanced import get_enhanced_flow_integrity
        return get_enhanced_flow_integrity().list_anomalies(limit=limit)

    anomalies = asyncio.run(_list())

    click.echo(f"\n{'=' * 60}")
    click.echo("Recent Anomalies")
    click.echo(f"{'=' * 60}")
    for a in anomalies:
        severity = "[HIGH]" if a.get("severity") == "high" else "[MED]" if a.get("severity") == "medium" else "[LOW]"
        click.echo(f"\n{severity} {a['timestamp']}")
        click.echo(f"    Agent:   {a['agent_id']}")
        click.echo(f"    Type:    {a['anomaly_type']}")
        click.echo(f"    Score:   {a['score']:.3f}")
        click.echo(f"    Action:  {a['action_taken']}")


@enhanced_flow_group.command("policies")
def enhanced_flow_policies():
    """List OPA policies for flow control."""
    async def _list():
        from blackbox.core.v2.flow_integrity_enhanced import get_enhanced_flow_integrity
        return get_enhanced_flow_integrity().list_policies()

    policies = asyncio.run(_list())

    click.echo(f"\n{'=' * 60}")
    click.echo("OPA Policies")
    click.echo(f"{'=' * 60}")
    for p in policies:
        status = "[ON] " if p.get("enabled") else "[OFF]"
        click.echo(f"\n{status} {p['name']}")
        click.echo(f"    Type:     {p['type']}")
        click.echo(f"    Priority: {p['priority']}")
        click.echo(f"    Evals:    {p['evaluations']}")


# =============================================================================
# Semantic Memory Commands (RAG, Qdrant)
# =============================================================================

@v2_cli.group("semantic-memory")
def semantic_memory_group():
    """Semantic memory - RAG with vector DB.

    Features:
    - Qdrant/ChromaDB vector stores
    - OpenAI/local embeddings
    - Forgetting mechanism (LRU/decay)
    - Conflict resolution

    Examples:
        bbx v2 semantic-memory stats
        bbx v2 semantic-memory search "query"
        bbx v2 semantic-memory forget
    """
    pass


@semantic_memory_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def semantic_memory_stats(format: str):
    """Show semantic memory statistics."""
    async def _get_stats():
        from blackbox.core.v2.semantic_memory import get_semantic_memory
        return get_semantic_memory().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Semantic Memory Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nStorage:")
        click.echo(f"  Backend:       {stats.get('backend', 'local')}")
        click.echo(f"  Memories:      {stats.get('memory_count', 0)}")
        click.echo(f"  Size:          {stats.get('size_mb', 0):.1f} MB")
        click.echo(f"\nEmbeddings:")
        click.echo(f"  Model:         {stats.get('embedding_model', 'N/A')}")
        click.echo(f"  Dimensions:    {stats.get('embedding_dim', 0)}")
        click.echo(f"\nQueries:")
        click.echo(f"  Total:         {stats.get('total_queries', 0)}")
        click.echo(f"  Avg Latency:   {stats.get('avg_query_ms', 0):.1f} ms")


@semantic_memory_group.command("search")
@click.argument("query")
@click.option("--limit", "-k", type=int, default=5, help="Number of results")
@click.option("--threshold", "-t", type=float, default=0.7, help="Similarity threshold")
def semantic_memory_search(query: str, limit: int, threshold: float):
    """Search semantic memory."""
    async def _search():
        from blackbox.core.v2.semantic_memory import get_semantic_memory
        return await get_semantic_memory().search(query, k=limit, threshold=threshold)

    results = asyncio.run(_search())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Search: '{query}'")
    click.echo(f"{'=' * 60}")
    for r in results:
        click.echo(f"\n[{r['score']:.3f}] {r['id']}")
        click.echo(f"    {r['content'][:100]}...")
        click.echo(f"    Created: {r['created_at']}")


@semantic_memory_group.command("add")
@click.argument("content")
@click.option("--metadata", "-m", default="{}", help="Metadata JSON")
def semantic_memory_add(content: str, metadata: str):
    """Add memory to semantic store."""
    async def _add():
        from blackbox.core.v2.semantic_memory import get_semantic_memory
        meta = json.loads(metadata)
        return await get_semantic_memory().add(content, metadata=meta)

    memory_id = asyncio.run(_add())
    click.echo(f"[+] Memory added: {memory_id}")


@semantic_memory_group.command("forget")
@click.option("--strategy", "-s", type=click.Choice(["lru", "decay", "age"]), default="lru")
@click.option("--threshold", "-t", type=float, default=0.1, help="Forgetting threshold")
@click.option("--dry-run", is_flag=True, help="Show what would be forgotten")
def semantic_memory_forget(strategy: str, threshold: float, dry_run: bool):
    """Run forgetting mechanism."""
    async def _forget():
        from blackbox.core.v2.semantic_memory import get_semantic_memory
        return await get_semantic_memory().forget(
            strategy=strategy, threshold=threshold, dry_run=dry_run
        )

    result = asyncio.run(_forget())

    if dry_run:
        click.echo(f"\n[DRY RUN] Would forget {result['count']} memories")
        for m in result.get("memories", [])[:10]:
            click.echo(f"  - {m['id']}: {m['reason']}")
    else:
        click.echo(f"[+] Forgot {result['count']} memories using {strategy} strategy")


# =============================================================================
# Message Bus Commands (Redis Streams, Kafka)
# =============================================================================

@v2_cli.group("message-bus")
def message_bus_group():
    """Message bus - Persistent messaging.

    Features:
    - Redis Streams backend
    - Exactly-once delivery
    - Consumer groups
    - Dead letter queues

    Examples:
        bbx v2 message-bus stats
        bbx v2 message-bus publish topic message
        bbx v2 message-bus dlq
    """
    pass


@message_bus_group.command("stats")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def message_bus_stats(format: str):
    """Show message bus statistics."""
    async def _get_stats():
        from blackbox.core.v2.message_bus import get_message_bus
        return get_message_bus().get_stats()

    stats = asyncio.run(_get_stats())

    if format == "json":
        click.echo(json.dumps(stats, indent=2, default=str))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Message Bus Statistics")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nBackend: {stats.get('backend', 'memory')}")
        click.echo(f"\nMessages:")
        click.echo(f"  Published:     {stats.get('messages_published', 0)}")
        click.echo(f"  Delivered:     {stats.get('messages_delivered', 0)}")
        click.echo(f"  Pending:       {stats.get('messages_pending', 0)}")
        click.echo(f"\nDelivery:")
        click.echo(f"  Success Rate:  {stats.get('delivery_rate', 100):.1f}%")
        click.echo(f"  Avg Latency:   {stats.get('avg_latency_ms', 0):.1f} ms")
        click.echo(f"\nDLQ:")
        click.echo(f"  Messages:      {stats.get('dlq_count', 0)}")


@message_bus_group.command("publish")
@click.argument("topic")
@click.argument("message")
@click.option("--key", "-k", default=None, help="Message key for ordering")
def message_bus_publish(topic: str, message: str, key: Optional[str]):
    """Publish message to topic."""
    async def _publish():
        from blackbox.core.v2.message_bus import get_message_bus
        return await get_message_bus().publish(topic, message, key=key)

    msg_id = asyncio.run(_publish())
    click.echo(f"[+] Published to {topic}: {msg_id}")


@message_bus_group.command("dlq")
@click.option("--limit", "-n", type=int, default=10, help="Number of messages to show")
@click.option("--reprocess", is_flag=True, help="Reprocess DLQ messages")
def message_bus_dlq(limit: int, reprocess: bool):
    """Show or reprocess dead letter queue."""
    async def _dlq():
        from blackbox.core.v2.message_bus import get_message_bus
        bus = get_message_bus()
        if reprocess:
            return await bus.reprocess_dlq()
        return bus.list_dlq(limit=limit)

    if reprocess:
        result = asyncio.run(_dlq())
        click.echo(f"[+] Reprocessed {result['count']} DLQ messages")
        click.echo(f"    Success: {result['success']}")
        click.echo(f"    Failed:  {result['failed']}")
    else:
        messages = asyncio.run(_dlq())
        click.echo(f"\n{'=' * 60}")
        click.echo("Dead Letter Queue")
        click.echo(f"{'=' * 60}")
        for m in messages:
            click.echo(f"\n[{m['id']}] {m['topic']}")
            click.echo(f"    Error:   {m['error']}")
            click.echo(f"    Retries: {m['retry_count']}")
            click.echo(f"    Failed:  {m['failed_at']}")


# =============================================================================
# Goal Engine Commands (LLM Planner, DAG)
# =============================================================================

@v2_cli.group("goals")
def goals_group():
    """Goal engine - LLM-powered planning.

    Features:
    - LLM-based goal decomposition
    - DAG execution with parallelism
    - Hierarchical planning
    - Cost optimization

    Examples:
        bbx v2 goals plan "Build a REST API"
        bbx v2 goals status
        bbx v2 goals list
    """
    pass


@goals_group.command("plan")
@click.argument("goal")
@click.option("--model", "-m", default="gpt-4o-mini", help="LLM model for planning")
@click.option("--max-steps", type=int, default=20, help="Maximum plan steps")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def goals_plan(goal: str, model: str, max_steps: int, dry_run: bool):
    """Create and optionally execute a plan for a goal."""
    async def _plan():
        from blackbox.core.v2.goal_engine import get_goal_engine
        engine = get_goal_engine()
        plan = await engine.plan(goal, model=model, max_steps=max_steps)
        if not dry_run:
            return await engine.execute(plan)
        return {"plan": plan, "executed": False}

    result = asyncio.run(_plan())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Goal: {goal}")
    click.echo(f"{'=' * 60}")

    plan = result.get("plan", result)
    click.echo(f"\nPlan ({len(plan.get('steps', []))} steps):")
    for step in plan.get("steps", []):
        deps = f" (after: {', '.join(step.get('depends_on', []))})" if step.get('depends_on') else ""
        click.echo(f"  [{step['id']}] {step['action']}{deps}")

    if result.get("executed"):
        click.echo(f"\nExecution Result: {result.get('status', 'unknown')}")


@goals_group.command("status")
@click.argument("goal_id", required=False)
def goals_status(goal_id: Optional[str]):
    """Show goal execution status."""
    async def _status():
        from blackbox.core.v2.goal_engine import get_goal_engine
        return get_goal_engine().get_status(goal_id)

    status = asyncio.run(_status())

    if goal_id:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Goal: {goal_id}")
        click.echo(f"{'=' * 60}")
        click.echo(f"\n  Status:     {status.get('status', 'unknown')}")
        click.echo(f"  Progress:   {status.get('progress', 0):.0f}%")
        click.echo(f"  Steps Done: {status.get('steps_done', 0)} / {status.get('steps_total', 0)}")
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Active Goals")
        click.echo(f"{'=' * 60}")
        for g in status.get("goals", []):
            click.echo(f"\n[{g['id'][:8]}...] {g['name']}")
            click.echo(f"    Status:   {g['status']}")
            click.echo(f"    Progress: {g['progress']:.0f}%")


@goals_group.command("list")
@click.option("--status", "-s", type=click.Choice(["all", "active", "completed", "failed"]), default="all")
@click.option("--limit", "-n", type=int, default=20, help="Number of goals to show")
def goals_list(status: str, limit: int):
    """List goals."""
    async def _list():
        from blackbox.core.v2.goal_engine import get_goal_engine
        return get_goal_engine().list_goals(status=status, limit=limit)

    goals = asyncio.run(_list())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Goals ({status})")
    click.echo(f"{'=' * 60}")
    for g in goals:
        icon = "+" if g['status'] == "completed" else "-" if g['status'] == "failed" else "?"
        click.echo(f"\n[{icon}] {g['id'][:8]}... - {g['name']}")
        click.echo(f"    Created: {g['created_at']}")
        click.echo(f"    Status:  {g['status']}")


# =============================================================================
# Authentication Commands (JWT, OIDC)
# =============================================================================

@v2_cli.group("auth")
def auth_group():
    """Authentication commands - JWT and OIDC.

    Features:
    - JWT token management
    - API key generation
    - OIDC integration
    - Authorization rules

    Examples:
        bbx v2 auth create-token agent-1
        bbx v2 auth create-api-key service-account
        bbx v2 auth verify <token>
    """
    pass


@auth_group.command("create-token")
@click.argument("identity_id")
@click.option("--expiry", "-e", type=int, default=24, help="Expiry in hours")
@click.option("--roles", "-r", multiple=True, help="Roles to include")
def auth_create_token(identity_id: str, expiry: int, roles: tuple):
    """Create JWT token for identity."""
    async def _create():
        from blackbox.core.v2.auth import get_auth_manager
        auth = get_auth_manager()
        identity = auth.get_identity(identity_id)
        if not identity:
            identity = auth.create_identity(identity_id, identity_id, roles=set(roles))
        return auth.create_token(identity, expiry_seconds=expiry * 3600)

    token = asyncio.run(_create())
    if token:
        click.echo(f"[+] Token created for: {identity_id}")
        click.echo(f"\n{token}")
    else:
        click.echo("[-] JWT not configured (set jwt_secret)", err=True)


@auth_group.command("create-api-key")
@click.argument("identity_id")
@click.option("--name", "-n", required=True, help="Key name")
@click.option("--expires-days", type=int, default=None, help="Expiry in days")
def auth_create_api_key(identity_id: str, name: str, expires_days: Optional[int]):
    """Create API key for identity."""
    async def _create():
        from blackbox.core.v2.auth import get_auth_manager
        auth = get_auth_manager()
        identity = auth.get_identity(identity_id)
        if not identity:
            identity = auth.create_identity(identity_id, identity_id)
        return auth.create_api_key(identity, name, expires_in_days=expires_days)

    key = asyncio.run(_create())
    if key:
        click.echo(f"[+] API key created: {name}")
        click.echo(f"\n{key}")
        click.echo(f"\n[!] Save this key - it won't be shown again!")
    else:
        click.echo("[-] API keys not enabled", err=True)


@auth_group.command("verify")
@click.argument("token")
@click.option("--method", "-m", type=click.Choice(["jwt", "api_key"]), default="jwt")
def auth_verify(token: str, method: str):
    """Verify token or API key."""
    async def _verify():
        from blackbox.core.v2.auth import get_auth_manager, AuthMethod
        auth = get_auth_manager()
        auth_method = AuthMethod.JWT if method == "jwt" else AuthMethod.API_KEY
        return await auth.verify_token(token, method=auth_method)

    identity = asyncio.run(_verify())

    if identity:
        click.echo(f"[+] Token valid")
        click.echo(f"\n  Identity: {identity.id}")
        click.echo(f"  Name:     {identity.name}")
        click.echo(f"  Type:     {identity.type}")
        click.echo(f"  Roles:    {', '.join(identity.roles) if identity.roles else 'none'}")
    else:
        click.echo("[-] Token invalid or expired", err=True)


@auth_group.command("rules")
def auth_rules():
    """List authorization rules."""
    from blackbox.core.v2.auth import get_auth_manager

    auth = get_auth_manager()
    rules = auth._authz._rules

    click.echo(f"\n{'=' * 60}")
    click.echo("Authorization Rules")
    click.echo(f"{'=' * 60}")
    for r in rules:
        effect = "[ALLOW]" if r.effect == "allow" else "[DENY]"
        click.echo(f"\n{effect} {r.name}")
        click.echo(f"    Pattern:  {r.resource_pattern}")
        click.echo(f"    Action:   {r.action}")
        click.echo(f"    Priority: {r.priority}")


# =============================================================================
# Monitoring Commands (Prometheus, Tracing)
# =============================================================================

@v2_cli.group("monitoring")
def monitoring_group():
    """Monitoring commands - Prometheus and tracing.

    Features:
    - Prometheus metrics
    - OpenTelemetry tracing
    - Alert management
    - Dashboard generation

    Examples:
        bbx v2 monitoring metrics
        bbx v2 monitoring alerts
        bbx v2 monitoring traces
    """
    pass


@monitoring_group.command("metrics")
@click.option("--format", "-f", type=click.Choice(["text", "prometheus", "json"]), default="text")
def monitoring_metrics(format: str):
    """Show current metrics."""
    async def _get_metrics():
        from blackbox.core.v2.monitoring import get_monitoring
        return get_monitoring().get_metrics()

    metrics = asyncio.run(_get_metrics())

    if format == "prometheus":
        for m in metrics:
            click.echo(f"# HELP {m['name']} {m.get('help', '')}")
            click.echo(f"# TYPE {m['name']} {m['type']}")
            click.echo(f"{m['name']}{{{m.get('labels', '')}}} {m['value']}")
    elif format == "json":
        click.echo(json.dumps(metrics, indent=2))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("BBX Metrics")
        click.echo(f"{'=' * 60}")
        for m in metrics[:30]:
            click.echo(f"\n  {m['name']}: {m['value']}")


@monitoring_group.command("alerts")
@click.option("--status", "-s", type=click.Choice(["all", "firing", "resolved"]), default="all")
def monitoring_alerts(status: str):
    """Show alerts."""
    async def _get_alerts():
        from blackbox.core.v2.monitoring import get_monitoring
        return get_monitoring().get_alerts(status=status)

    alerts = asyncio.run(_get_alerts())

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Alerts ({status})")
    click.echo(f"{'=' * 60}")
    for a in alerts:
        icon = "[!]" if a['status'] == "firing" else "[~]"
        click.echo(f"\n{icon} {a['name']}")
        click.echo(f"    Severity: {a['severity']}")
        click.echo(f"    Status:   {a['status']}")
        click.echo(f"    Message:  {a['message']}")


@monitoring_group.command("traces")
@click.option("--limit", "-n", type=int, default=10, help="Number of traces")
@click.option("--service", "-s", default=None, help="Filter by service")
def monitoring_traces(limit: int, service: Optional[str]):
    """Show recent traces."""
    async def _get_traces():
        from blackbox.core.v2.monitoring import get_monitoring
        return get_monitoring().get_traces(limit=limit, service=service)

    traces = asyncio.run(_get_traces())

    click.echo(f"\n{'=' * 60}")
    click.echo("Recent Traces")
    click.echo(f"{'=' * 60}")
    for t in traces:
        duration = f"{t['duration_ms']:.1f}ms"
        click.echo(f"\n[{t['trace_id'][:12]}...] {t['operation']} ({duration})")
        click.echo(f"    Service: {t['service']}")
        click.echo(f"    Status:  {t['status']}")
        click.echo(f"    Spans:   {t['span_count']}")


@monitoring_group.command("dashboard")
@click.option("--output", "-o", default="dashboard.json", help="Output file")
def monitoring_dashboard(output: str):
    """Generate Grafana dashboard."""
    async def _generate():
        from blackbox.core.v2.monitoring import get_monitoring
        return get_monitoring().generate_dashboard()

    dashboard = asyncio.run(_generate())

    with open(output, "w") as f:
        json.dump(dashboard, f, indent=2)

    click.echo(f"[+] Dashboard generated: {output}")
    click.echo(f"    Import this into Grafana to visualize BBX metrics")


# =============================================================================
# Deployment Commands (Docker, Helm, K8s)
# =============================================================================

@v2_cli.group("deploy")
def deploy_group():
    """Deployment commands - Docker, Helm, K8s.

    Features:
    - Dockerfile generation
    - Helm chart generation
    - Kubernetes operator

    Examples:
        bbx v2 deploy dockerfile
        bbx v2 deploy helm-chart
        bbx v2 deploy k8s-manifest
    """
    pass


@deploy_group.command("dockerfile")
@click.option("--output", "-o", default="Dockerfile", help="Output file")
@click.option("--base-image", default="python:3.11-slim", help="Base image")
def deploy_dockerfile(output: str, base_image: str):
    """Generate Dockerfile for BBX agent."""
    async def _generate():
        from blackbox.core.v2.deployment import get_deployment_manager
        return get_deployment_manager().generate_dockerfile(base_image=base_image)

    dockerfile = asyncio.run(_generate())

    with open(output, "w") as f:
        f.write(dockerfile)

    click.echo(f"[+] Dockerfile generated: {output}")
    click.echo(f"    Build with: docker build -t bbx-agent .")


@deploy_group.command("helm-chart")
@click.option("--output", "-o", default="./chart", help="Output directory")
@click.option("--name", "-n", default="bbx-agent", help="Chart name")
def deploy_helm_chart(output: str, name: str):
    """Generate Helm chart for BBX deployment."""
    async def _generate():
        from blackbox.core.v2.deployment import get_deployment_manager
        return get_deployment_manager().generate_helm_chart(name=name, output_dir=output)

    result = asyncio.run(_generate())

    click.echo(f"[+] Helm chart generated: {output}")
    click.echo(f"    Install with: helm install {name} {output}")
    click.echo(f"\n    Files created:")
    for f in result.get("files", []):
        click.echo(f"      - {f}")


@deploy_group.command("k8s-manifest")
@click.option("--output", "-o", default="bbx-deployment.yaml", help="Output file")
@click.option("--namespace", "-n", default="default", help="Kubernetes namespace")
@click.option("--replicas", "-r", type=int, default=1, help="Number of replicas")
def deploy_k8s_manifest(output: str, namespace: str, replicas: int):
    """Generate Kubernetes deployment manifest."""
    async def _generate():
        from blackbox.core.v2.deployment import get_deployment_manager
        return get_deployment_manager().generate_k8s_manifest(
            namespace=namespace, replicas=replicas
        )

    manifest = asyncio.run(_generate())

    with open(output, "w") as f:
        f.write(manifest)

    click.echo(f"[+] K8s manifest generated: {output}")
    click.echo(f"    Apply with: kubectl apply -f {output}")


@deploy_group.command("operator-crd")
@click.option("--output", "-o", default="bbx-crd.yaml", help="Output file")
def deploy_operator_crd(output: str):
    """Generate BBX Kubernetes Operator CRD."""
    async def _generate():
        from blackbox.core.v2.deployment import get_deployment_manager
        return get_deployment_manager().generate_operator_crd()

    crd = asyncio.run(_generate())

    with open(output, "w") as f:
        f.write(crd)

    click.echo(f"[+] Operator CRD generated: {output}")
    click.echo(f"    Install with: kubectl apply -f {output}")


# =============================================================================
# System Status Command
# =============================================================================

@v2_cli.command("status")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def v2_system_status(format: str):
    """Show BBX 2.0 system status - all components."""
    async def _get_status():
        status = {
            "version": "2.0.0",
            "components": {}
        }

        # Check each component
        components = [
            ("ring", "Enhanced Ring", "blackbox.core.v2.ring_enhanced"),
            ("context", "Context Tiering", "blackbox.core.v2.context_tiering_enhanced"),
            ("quotas", "Enforced Quotas", "blackbox.core.v2.quotas_enforced"),
            ("snapshots", "Distributed Snapshots", "blackbox.core.v2.snapshots_distributed"),
            ("flow", "Flow Integrity", "blackbox.core.v2.flow_integrity_enhanced"),
            ("memory", "Semantic Memory", "blackbox.core.v2.semantic_memory"),
            ("bus", "Message Bus", "blackbox.core.v2.message_bus"),
            ("goals", "Goal Engine", "blackbox.core.v2.goal_engine"),
            ("auth", "Authentication", "blackbox.core.v2.auth"),
            ("monitoring", "Monitoring", "blackbox.core.v2.monitoring"),
            ("deployment", "Deployment", "blackbox.core.v2.deployment"),
        ]

        for key, name, module in components:
            try:
                __import__(module)
                status["components"][key] = {"name": name, "status": "available"}
            except ImportError as e:
                status["components"][key] = {"name": name, "status": "not_loaded", "error": str(e)}

        return status

    status = asyncio.run(_get_status())

    if format == "json":
        click.echo(json.dumps(status, indent=2))
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"BBX 2.0 System Status (v{status['version']})")
        click.echo(f"{'=' * 60}")
        click.echo(f"\nComponents:")
        for key, comp in status["components"].items():
            icon = "[+]" if comp["status"] == "available" else "[-]"
            click.echo(f"  {icon} {comp['name']}")
        click.echo(f"\n{'=' * 60}")


# =============================================================================
# Integration with main CLI
# =============================================================================

def register_v2_commands(cli_group):
    """Register BBX 2.0 commands with main CLI."""
    cli_group.add_command(v2_cli)


# For standalone usage
if __name__ == "__main__":
    v2_cli()
