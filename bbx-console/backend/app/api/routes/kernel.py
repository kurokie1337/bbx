"""
BBX Kernel API Routes

API endpoints for BBX Kernel development environment:
- File browsing and reading
- Build operations (cargo build)
- QEMU launcher
- Syscall reference
"""

import os
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/kernel", tags=["kernel"])

# Kernel directory path (relative to project root)
KERNEL_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "kernel"


class FileResponse(BaseModel):
    content: str
    path: str
    language: str


class BuildResponse(BaseModel):
    success: bool
    output: List[str]
    error: Optional[str] = None


class KernelInfo(BaseModel):
    path: str
    exists: bool
    files_count: int
    rust_files_count: int
    has_cargo_toml: bool


# ============================================================================
# File Operations
# ============================================================================

@router.get("/file", response_model=FileResponse)
async def get_kernel_file(path: str = Query(..., description="Relative path to kernel file")):
    """
    Get contents of a kernel source file.
    Path should be relative to kernel directory (e.g., 'kernel/src/main.rs')
    """
    # Remove 'kernel/' prefix if present
    if path.startswith("kernel/"):
        path = path[7:]

    file_path = KERNEL_DIR / path

    # Security: ensure file is within kernel directory
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(KERNEL_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied: path outside kernel directory")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    # Determine language for syntax highlighting
    suffix = file_path.suffix.lower()
    language_map = {
        ".rs": "rust",
        ".toml": "toml",
        ".md": "markdown",
        ".s": "asm",
        ".asm": "asm",
        ".ld": "linker",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    language = language_map.get(suffix, "text")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    return FileResponse(
        content=content,
        path=str(path),
        language=language,
    )


@router.get("/info", response_model=KernelInfo)
async def get_kernel_info():
    """Get information about the kernel directory"""
    if not KERNEL_DIR.exists():
        return KernelInfo(
            path=str(KERNEL_DIR),
            exists=False,
            files_count=0,
            rust_files_count=0,
            has_cargo_toml=False,
        )

    # Count files
    all_files = list(KERNEL_DIR.rglob("*"))
    files_count = len([f for f in all_files if f.is_file()])
    rust_files_count = len([f for f in all_files if f.suffix == ".rs"])
    has_cargo_toml = (KERNEL_DIR / "Cargo.toml").exists()

    return KernelInfo(
        path=str(KERNEL_DIR),
        exists=True,
        files_count=files_count,
        rust_files_count=rust_files_count,
        has_cargo_toml=has_cargo_toml,
    )


# ============================================================================
# Build Operations
# ============================================================================

@router.post("/build", response_model=BuildResponse)
async def build_kernel():
    """
    Build the BBX Kernel using cargo.
    Requires Rust nightly toolchain with rust-src and llvm-tools-preview components.
    """
    if not KERNEL_DIR.exists():
        raise HTTPException(status_code=404, detail="Kernel directory not found")

    if not (KERNEL_DIR / "Cargo.toml").exists():
        raise HTTPException(status_code=400, detail="Cargo.toml not found in kernel directory")

    output_lines = []

    try:
        # Run cargo build
        process = await asyncio.create_subprocess_exec(
            "cargo", "build", "--release",
            cwd=str(KERNEL_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=300)  # 5 min timeout

        output_text = stdout.decode("utf-8", errors="replace")
        output_lines = output_text.split("\n")

        if process.returncode == 0:
            return BuildResponse(
                success=True,
                output=output_lines,
            )
        else:
            return BuildResponse(
                success=False,
                output=output_lines,
                error=f"Build failed with exit code {process.returncode}",
            )

    except asyncio.TimeoutError:
        return BuildResponse(
            success=False,
            output=output_lines + ["Build timed out after 5 minutes"],
            error="Build timeout",
        )
    except FileNotFoundError:
        return BuildResponse(
            success=False,
            output=["cargo: command not found"],
            error="Rust/Cargo not installed. Install from https://rustup.rs/",
        )
    except Exception as e:
        return BuildResponse(
            success=False,
            output=output_lines + [str(e)],
            error=str(e),
        )


@router.post("/bootimage", response_model=BuildResponse)
async def create_bootimage():
    """
    Create bootable kernel image using cargo bootimage.
    Requires bootimage tool: cargo install bootimage
    """
    if not KERNEL_DIR.exists():
        raise HTTPException(status_code=404, detail="Kernel directory not found")

    output_lines = []

    try:
        # Run cargo bootimage
        process = await asyncio.create_subprocess_exec(
            "cargo", "bootimage", "--release",
            cwd=str(KERNEL_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=600)  # 10 min timeout

        output_text = stdout.decode("utf-8", errors="replace")
        output_lines = output_text.split("\n")

        if process.returncode == 0:
            return BuildResponse(
                success=True,
                output=output_lines,
            )
        else:
            return BuildResponse(
                success=False,
                output=output_lines,
                error=f"Bootimage creation failed with exit code {process.returncode}",
            )

    except asyncio.TimeoutError:
        return BuildResponse(
            success=False,
            output=output_lines + ["Bootimage creation timed out after 10 minutes"],
            error="Bootimage timeout",
        )
    except FileNotFoundError:
        return BuildResponse(
            success=False,
            output=["cargo bootimage: command not found"],
            error="bootimage tool not installed. Run: cargo install bootimage",
        )
    except Exception as e:
        return BuildResponse(
            success=False,
            output=output_lines + [str(e)],
            error=str(e),
        )


# ============================================================================
# QEMU Operations
# ============================================================================

@router.post("/qemu")
async def run_qemu():
    """
    Run the kernel in QEMU emulator.
    Requires QEMU: qemu-system-x86_64
    """
    bootimage_path = KERNEL_DIR / "target" / "x86_64-unknown-none" / "release" / "bootimage-bbx-kernel.bin"

    if not bootimage_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Bootimage not found. Run /kernel/bootimage first to create it."
        )

    try:
        # Start QEMU in background
        process = await asyncio.create_subprocess_exec(
            "qemu-system-x86_64",
            "-drive", f"format=raw,file={bootimage_path}",
            "-serial", "stdio",
            "-no-reboot",
            "-no-shutdown",
            cwd=str(KERNEL_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Note: In a real implementation, we'd want to stream this output
        # via WebSocket for real-time updates

        return {
            "success": True,
            "message": "QEMU started",
            "pid": process.pid,
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="QEMU not installed. Install qemu-system-x86_64."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Syscall Reference
# ============================================================================

@router.get("/syscalls")
async def get_syscalls():
    """Get BBX Kernel syscall reference"""
    return {
        "syscalls": [
            # Process/Task Management
            {"num": 0, "name": "spawn", "category": "process", "desc": "Create new task"},
            {"num": 1, "name": "exit", "category": "process", "desc": "Exit current task"},
            {"num": 2, "name": "getpid", "category": "process", "desc": "Get current task ID"},
            {"num": 3, "name": "wait", "category": "process", "desc": "Wait for task completion"},
            {"num": 4, "name": "kill", "category": "process", "desc": "Kill task"},
            {"num": 5, "name": "get_task_info", "category": "process", "desc": "Get task info (like ps)"},

            # Memory Management
            {"num": 10, "name": "mmap", "category": "memory", "desc": "Allocate memory"},
            {"num": 11, "name": "munmap", "category": "memory", "desc": "Free memory"},
            {"num": 12, "name": "mem_info", "category": "memory", "desc": "Get memory info"},

            # I/O Ring (AgentRing)
            {"num": 20, "name": "io_submit", "category": "ring", "desc": "Submit I/O operation to ring"},
            {"num": 21, "name": "io_wait", "category": "ring", "desc": "Wait for I/O completion"},
            {"num": 22, "name": "io_submit_batch", "category": "ring", "desc": "Submit batch of I/O operations"},
            {"num": 23, "name": "io_cancel", "category": "ring", "desc": "Cancel I/O operation"},

            # State/Storage
            {"num": 30, "name": "state_get", "category": "state", "desc": "Get state value"},
            {"num": 31, "name": "state_set", "category": "state", "desc": "Set state value"},
            {"num": 32, "name": "state_del", "category": "state", "desc": "Delete state value"},
            {"num": 33, "name": "state_list", "category": "state", "desc": "List state keys"},

            # Agent Communication (A2A)
            {"num": 40, "name": "agent_send", "category": "a2a", "desc": "Send message to agent"},
            {"num": 41, "name": "agent_recv", "category": "a2a", "desc": "Receive message"},
            {"num": 42, "name": "agent_call", "category": "a2a", "desc": "Call agent skill"},
            {"num": 43, "name": "agent_discover", "category": "a2a", "desc": "Discover agents"},

            # Workflow Execution
            {"num": 50, "name": "workflow_run", "category": "workflow", "desc": "Run workflow"},
            {"num": 51, "name": "workflow_status", "category": "workflow", "desc": "Get workflow status"},
            {"num": 52, "name": "workflow_cancel", "category": "workflow", "desc": "Cancel workflow"},

            # System Info
            {"num": 60, "name": "time", "category": "system", "desc": "Get system time"},
            {"num": 61, "name": "uptime", "category": "system", "desc": "Get uptime"},
            {"num": 62, "name": "sys_info", "category": "system", "desc": "Get system info"},

            # Console I/O
            {"num": 70, "name": "write", "category": "console", "desc": "Write to console"},
            {"num": 71, "name": "read", "category": "console", "desc": "Read from console"},

            # Misc
            {"num": 100, "name": "yield", "category": "misc", "desc": "Yield CPU"},
            {"num": 101, "name": "sleep", "category": "misc", "desc": "Sleep for ms"},
        ]
    }
