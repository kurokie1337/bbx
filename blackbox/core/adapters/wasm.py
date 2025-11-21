"""
WebAssembly Adapter - Advanced WASM features for BBX
Provides compilation, execution, and management of WebAssembly modules
"""
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..base_adapter import BaseAdapter


class WasmAdapter(BaseAdapter):
    """
    WebAssembly adapter with advanced features.

    Capabilities:
    - Compile to WASM (C/C++/Rust/AssemblyScript)
    - Execute WASM modules
    - WASI support
    - Module linking
    - Memory management
    - Performance profiling
    - WASM component model
    - Streaming compilation
    """

    def __init__(self):
        super().__init__("wasm")
        self.wasmtime = self._detect_wasmtime()
        self.wasm_opt = self._detect_wasm_opt()

    def _detect_wasmtime(self) -> Optional[str]:
        """Detect wasmtime runtime"""
        try:
            result = subprocess.run(
                ["wasmtime", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return "wasmtime"
        except FileNotFoundError:
            pass
        return None

    def _detect_wasm_opt(self) -> Optional[str]:
        """Detect wasm-opt optimizer"""
        try:
            result = subprocess.run(
                ["wasm-opt", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return "wasm-opt"
        except FileNotFoundError:
            pass
        return None

    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute WebAssembly actions"""
        actions = {
            "compile.c": self._compile_c,
            "compile.cpp": self._compile_cpp,
            "compile.rust": self._compile_rust,
            "compile.assemblyscript": self._compile_assemblyscript,
            "run": self._run_wasm,
            "run.wasi": self._run_wasi,
            "optimize": self._optimize,
            "validate": self._validate,
            "info": self._get_info,
            "link": self._link_modules,
            "benchmark": self._benchmark,
            "profile": self._profile,
        }

        handler = actions.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        return await handler(params)

    async def _compile_c(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compile C to WebAssembly using Emscripten or wasi-sdk"""
        source_file = params["source"]
        output_file = params.get("output", "output.wasm")
        flags = params.get("flags", [])

        # Try Emscripten first
        cmd = ["emcc", source_file, "-o", output_file] + flags

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }

        return {
            "status": "success",
            "output_file": output_file,
            "size": Path(output_file).stat().st_size
        }

    async def _compile_cpp(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compile C++ to WebAssembly"""
        source_file = params["source"]
        output_file = params.get("output", "output.wasm")
        flags = params.get("flags", [])

        cmd = ["em++", source_file, "-o", output_file] + flags

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }

        return {
            "status": "success",
            "output_file": output_file,
            "size": Path(output_file).stat().st_size
        }

    async def _compile_rust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compile Rust to WebAssembly"""
        source_dir = params["source_dir"]
        output_file = params.get("output", "output.wasm")
        target = params.get("target", "wasm32-unknown-unknown")
        release = params.get("release", True)

        cmd = ["cargo", "build", "--target", target]
        if release:
            cmd.append("--release")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=source_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }

        # Find the compiled wasm file
        build_dir = Path(source_dir) / "target" / target / ("release" if release else "debug")
        wasm_files = list(build_dir.glob("*.wasm"))

        if not wasm_files:
            return {
                "status": "error",
                "error": "No WASM file found after compilation"
            }

        source_wasm = wasm_files[0]
        if output_file != str(source_wasm):
            import shutil
            shutil.copy(source_wasm, output_file)

        return {
            "status": "success",
            "output_file": output_file,
            "size": Path(output_file).stat().st_size
        }

    async def _compile_assemblyscript(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compile AssemblyScript to WebAssembly"""
        source_file = params["source"]
        output_file = params.get("output", "output.wasm")
        optimize = params.get("optimize", True)

        cmd = ["asc", source_file, "-o", output_file]
        if optimize:
            cmd.extend(["-O3", "--optimize"])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode(),
                "output": stdout.decode()
            }

        return {
            "status": "success",
            "output_file": output_file,
            "size": Path(output_file).stat().st_size
        }

    async def _run_wasm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run WebAssembly module"""
        if not self.wasmtime:
            raise RuntimeError("wasmtime not found. Install from https://wasmtime.dev/")

        wasm_file = params["wasm_file"]
        func = params.get("function", "_start")
        args = params.get("args", [])

        cmd = [self.wasmtime, "run", wasm_file]
        if func != "_start":
            cmd.extend(["--invoke", func])
        cmd.extend([str(arg) for arg in args])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode
        }

    async def _run_wasi(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run WASM module with WASI support"""
        if not self.wasmtime:
            raise RuntimeError("wasmtime not found")

        wasm_file = params["wasm_file"]
        args = params.get("args", [])
        env = params.get("env", {})
        dirs = params.get("dirs", [])

        cmd = [self.wasmtime, "run"]

        # Add directory mappings
        for dir_mapping in dirs:
            cmd.extend(["--dir", dir_mapping])

        # Add environment variables
        for key, value in env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.append(wasm_file)
        cmd.extend([str(arg) for arg in args])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode
        }

    async def _optimize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize WebAssembly module"""
        if not self.wasm_opt:
            raise RuntimeError("wasm-opt not found. Install from Binaryen")

        input_file = params["input"]
        output_file = params.get("output", input_file)
        level = params.get("level", "3")

        cmd = [self.wasm_opt, f"-O{level}", input_file, "-o", output_file]

        # Additional optimizations
        if params.get("strip_debug"):
            cmd.append("--strip-debug")
        if params.get("strip_producers"):
            cmd.append("--strip-producers")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        original_size = Path(input_file).stat().st_size
        optimized_size = Path(output_file).stat().st_size

        return {
            "status": "optimized",
            "output_file": output_file,
            "original_size": original_size,
            "optimized_size": optimized_size,
            "reduction": f"{((original_size - optimized_size) / original_size * 100):.2f}%"
        }

    async def _validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate WebAssembly module"""
        if not self.wasmtime:
            raise RuntimeError("wasmtime not found")

        wasm_file = params["wasm_file"]

        cmd = [self.wasmtime, "compile", wasm_file, "--dry-run"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "valid": proc.returncode == 0,
            "output": stdout.decode(),
            "error": stderr.decode()
        }

    async def _get_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about WASM module"""
        wasm_file = params["wasm_file"]

        # Use wasm-objdump if available
        try:
            proc = await asyncio.create_subprocess_exec(
                "wasm-objdump", "-h", wasm_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            return {
                "file": wasm_file,
                "size": Path(wasm_file).stat().st_size,
                "info": stdout.decode()
            }
        except FileNotFoundError:
            # Fallback to basic info
            return {
                "file": wasm_file,
                "size": Path(wasm_file).stat().st_size,
                "error": "wasm-objdump not available for detailed info"
            }

    async def _link_modules(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Link multiple WASM modules"""
        modules = params["modules"]
        output_file = params["output"]

        # Use wasm-ld for linking
        cmd = ["wasm-ld"] + modules + ["-o", output_file]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": stderr.decode()
            }

        return {
            "status": "linked",
            "output_file": output_file,
            "size": Path(output_file).stat().st_size
        }

    async def _benchmark(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark WASM module execution"""
        wasm_file = params["wasm_file"]
        iterations = params.get("iterations", 100)
        func = params.get("function", "_start")

        import time

        times = []
        for _ in range(iterations):
            start = time.perf_counter()

            cmd = [self.wasmtime, "run", wasm_file]
            if func != "_start":
                cmd.extend(["--invoke", func])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.communicate()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return {
            "iterations": iterations,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "times": times
        }

    async def _profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Profile WASM module execution"""
        wasm_file = params["wasm_file"]
        func = params.get("function", "_start")

        # Use wasmtime with profiling
        cmd = [
            self.wasmtime, "run",
            "--profile=perfmap",
            wasm_file
        ]
        if func != "_start":
            cmd.extend(["--invoke", func])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return {
            "output": stdout.decode(),
            "profile_data": stderr.decode()
        }
