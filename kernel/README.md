# BBX Kernel

**Bare-metal Operating System Kernel for AI Agents**

```
    ____  ____  _  __   ____  _____
   | __ )| __ )\ \/ /  / __ \/ ___/
   |  _ \|  _ \ \  /  / / / /\__ \
   | |_) | |_) |/  \ / /_/ /___/ /
   |____/|____//_/\_\\____//____/

   Operating System for AI Agents
   Copyright 2025 Ilya Makarov
```

## Overview

BBX Kernel is a bare-metal operating system kernel designed specifically for AI agents. It brings BBX's high-level concepts (AgentRing, ContextTiering, DAG execution) to the hardware level.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER SPACE (Agents)                        │
│                   (BBX Workflows, A2A Protocol)                 │
├─────────────────────────────────────────────────────────────────┤
│                    SYSCALL INTERFACE                            │
│         (spawn, io_submit, state_get, agent_call, etc.)        │
├─────────────────────────────────────────────────────────────────┤
│                      BBX KERNEL                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  SCHEDULER  │  │   MEMORY    │  │       I/O RING          │ │
│  │  (DAG-based)│  │  (Tiered)   │  │  (io_uring-inspired)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ INTERRUPTS  │  │    CPU      │  │       DRIVERS           │ │
│  │   (IDT)     │  │   (GDT)     │  │ (serial, timer, kbd)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                       HARDWARE                                   │
│              (x86_64 CPU, RAM, Storage, Network)                │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Scheduler (`src/scheduler/`)
- **Task-based**: Each agent thought is a Task
- **Priority queues**: REALTIME > HIGH > NORMAL > LOW
- **DAG support**: Tasks can have dependencies
- **Preemptive**: Time-slice based scheduling

### Memory Manager (`src/memory/`)
- **Tiered memory**: HOT → WARM → COOL → COLD (like BBX ContextTiering)
- **Frame allocator**: Physical page management
- **Heap allocator**: Dynamic memory allocation
- **Paging**: Virtual memory support

### I/O Ring (`src/scheduler/ring.rs`)
- **io_uring-inspired**: Batch submission/completion
- **Priority-based**: Operations processed by priority
- **Zero-copy**: Where possible

### Syscalls (`src/syscall/`)
- **Process**: spawn, exit, wait, kill
- **Memory**: mmap, munmap
- **I/O**: io_submit, io_wait
- **State**: state_get, state_set
- **Agent**: agent_send, agent_call
- **Workflow**: workflow_run, workflow_status

### Drivers (`src/drivers/`)
- **Serial**: UART 16550 for debug output
- **Timer**: PIT for scheduling
- **Keyboard**: PS/2 keyboard input

## Building

### Prerequisites

```bash
# Install Rust nightly
rustup install nightly
rustup default nightly

# Add required components
rustup component add rust-src llvm-tools-preview

# Install bootimage tool
cargo install bootimage
```

### Build

```bash
cd kernel

# Build kernel
cargo build --release

# Create bootable image
cargo bootimage --release
```

### Run in QEMU

```bash
# Run with QEMU
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unknown-none/release/bootimage-bbx-kernel.bin

# With serial output to terminal
qemu-system-x86_64 \
    -drive format=raw,file=target/x86_64-unknown-none/release/bootimage-bbx-kernel.bin \
    -serial stdio
```

## Syscall Reference

| Number | Name | Description |
|--------|------|-------------|
| 0 | spawn | Create new task |
| 1 | exit | Exit current task |
| 2 | getpid | Get current task ID |
| 3 | wait | Wait for task |
| 4 | kill | Kill task |
| 20 | io_submit | Submit I/O to ring |
| 21 | io_wait | Wait for I/O completion |
| 30 | state_get | Get state value |
| 31 | state_set | Set state value |
| 40 | agent_send | Send to agent |
| 42 | agent_call | Call agent skill |
| 50 | workflow_run | Run workflow |
| 60 | time | Get system time |
| 61 | uptime | Get uptime |
| 100 | yield | Yield CPU |
| 101 | sleep | Sleep for ms |

## Integration with BBX

BBX Kernel is designed to run BBX Python runtime in user space:

```
┌─────────────────────────────────────────┐
│         BBX Python Runtime              │
│   (blackbox.core, adapters, A2A, etc.)  │
│                                         │
│   Uses syscalls to interact with kernel │
└────────────────┬────────────────────────┘
                 │ syscalls
┌────────────────▼────────────────────────┐
│           BBX Kernel                    │
│   (Memory, Scheduler, I/O Ring)         │
└─────────────────────────────────────────┘
```

## License

BSL-1.1 (converts to Apache 2.0 on 2028-11-05)

## Author

Ilya Makarov, Krasnoyarsk, Russia
