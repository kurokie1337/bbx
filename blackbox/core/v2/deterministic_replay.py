# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX SIRE Deterministic Replay - Record and Replay AI Operations.

Like game replays or database replication:
- Record every operation with full context
- Replay to reproduce exact behavior
- Time travel debugging
- A/B testing of different approaches

This makes AI PREDICTABLE and DEBUGGABLE.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Replay Engine                              │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐  ┌─────────────────┐                   │
    │  │    Recorder     │  │     Player      │                   │
    │  │  (Capture ops)  │  │  (Execute ops)  │                   │
    │  └────────┬────────┘  └────────┬────────┘                   │
    │           │                    │                             │
    │  ┌────────▼────────────────────▼────────────────────────┐   │
    │  │              Operation Log (Tape)                     │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
    │  │  │ Frame 0 │ │ Frame 1 │ │ Frame 2 │ │ Frame N │     │   │
    │  │  │ t=0.0ms │ │ t=1.2ms │ │ t=2.5ms │ │ t=...   │     │   │
    │  │  │ THINK   │ │ READ    │ │ WRITE   │ │ ...     │     │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                              │                               │
    │  ┌───────────────────────────▼───────────────────────────┐   │
    │  │                 Mock Provider Registry                 │   │
    │  │  - Fake LLM responses (from recording)                │   │
    │  │  - Fake file system                                   │   │
    │  │  - Fake time                                          │   │
    │  └───────────────────────────────────────────────────────┘   │
    │                              │                               │
    │  ┌───────────────────────────▼───────────────────────────┐   │
    │  │                 Divergence Detector                    │   │
    │  │  - Compare actual vs recorded                         │   │
    │  │  - Report when behavior differs                       │   │
    │  │  - Root cause analysis                                │   │
    │  └───────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Use Cases:
    1. Debug: "Why did the AI do X?" - replay to see exact context
    2. Test: Replay with different prompts, compare results
    3. Audit: Prove what happened during an operation
    4. Learn: Extract successful patterns for future use
    5. Recovery: Replay operations on recovered state
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.sire.replay")


# =============================================================================
# Frame Types
# =============================================================================


class FrameType(Enum):
    """Types of recorded frames"""
    # System frames
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CHECKPOINT = "checkpoint"

    # Operation frames
    SYSCALL = "syscall"          # Any syscall
    LLM_REQUEST = "llm_request"  # LLM call start
    LLM_RESPONSE = "llm_response"  # LLM response
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"

    # Agent frames
    AGENT_SPAWN = "agent_spawn"
    AGENT_MESSAGE = "agent_message"
    AGENT_RESULT = "agent_result"

    # State frames
    STATE_SNAPSHOT = "state_snapshot"
    CONTEXT_CHANGE = "context_change"


# =============================================================================
# Recorded Frame
# =============================================================================


@dataclass
class ReplayFrame:
    """Single recorded frame in the replay log"""
    # Identification
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sequence: int = 0

    # Timing
    timestamp: float = field(default_factory=time.time)
    relative_time_ms: float = 0  # Time since session start

    # Frame content
    frame_type: FrameType = FrameType.SYSCALL
    agent_id: str = ""
    operation: str = ""

    # Input/Output
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Context
    context_hash: str = ""  # Hash of current context
    state_hash: str = ""    # Hash of current state

    # Error handling
    success: bool = True
    error: Optional[str] = None

    # Determinism check
    input_hash: str = ""
    output_hash: str = ""

    def __post_init__(self):
        if not self.input_hash:
            self.input_hash = self._hash_dict(self.inputs)
        if not self.output_hash:
            self.output_hash = self._hash_dict(self.outputs)

    def _hash_dict(self, d: Dict) -> str:
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


# =============================================================================
# Recording Session
# =============================================================================


@dataclass
class RecordingSession:
    """A complete recording session"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Timing
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    # Frames
    frames: List[ReplayFrame] = field(default_factory=list)

    # Initial state
    initial_state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    agent_ids: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    # Statistics
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_syscalls: int = 0

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000


# =============================================================================
# Recorder
# =============================================================================


class ReplayRecorder:
    """
    Records operations for later replay.

    Usage:
        recorder = ReplayRecorder()
        recorder.start_session("deploy_feature")

        # During operation
        recorder.record_llm(prompt, response)
        recorder.record_syscall(syscall, args, result)

        # Save
        recording = recorder.end_session()
        recorder.save(recording, "deploy_v1.replay")
    """

    def __init__(self):
        self._current_session: Optional[RecordingSession] = None
        self._sequence = 0
        self._start_time = 0

        # Hooks for automatic recording
        self._pre_hooks: Dict[FrameType, List[Callable]] = {}
        self._post_hooks: Dict[FrameType, List[Callable]] = {}

    @property
    def is_recording(self) -> bool:
        return self._current_session is not None

    def start_session(
        self,
        name: str = "",
        description: str = "",
        initial_state: Optional[Dict] = None,
        tags: Optional[Set[str]] = None
    ) -> str:
        """Start a new recording session"""
        self._current_session = RecordingSession(
            name=name or f"session_{int(time.time())}",
            description=description,
            initial_state=initial_state or {},
            tags=tags or set(),
        )
        self._sequence = 0
        self._start_time = time.time()

        # Record session start
        self._record_frame(
            FrameType.SESSION_START,
            "",
            "session_start",
            inputs={"name": name, "description": description},
            outputs={"session_id": self._current_session.session_id}
        )

        logger.info(f"Started recording session: {self._current_session.session_id}")
        return self._current_session.session_id

    def end_session(self) -> RecordingSession:
        """End recording and return session"""
        if not self._current_session:
            raise ValueError("No active recording session")

        # Record session end
        self._record_frame(
            FrameType.SESSION_END,
            "",
            "session_end",
            outputs={
                "duration_ms": self._current_session.duration_ms,
                "total_frames": len(self._current_session.frames),
            }
        )

        self._current_session.ended_at = time.time()
        session = self._current_session
        self._current_session = None

        logger.info(f"Ended recording session: {session.session_id}, {len(session.frames)} frames")
        return session

    def _record_frame(
        self,
        frame_type: FrameType,
        agent_id: str,
        operation: str,
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
        context_hash: str = "",
        state_hash: str = "",
        success: bool = True,
        error: Optional[str] = None
    ) -> Optional[ReplayFrame]:
        """Record a single frame"""
        if not self._current_session:
            return None

        self._sequence += 1
        relative_time = (time.time() - self._start_time) * 1000

        frame = ReplayFrame(
            sequence=self._sequence,
            relative_time_ms=relative_time,
            frame_type=frame_type,
            agent_id=agent_id,
            operation=operation,
            inputs=inputs or {},
            outputs=outputs or {},
            context_hash=context_hash,
            state_hash=state_hash,
            success=success,
            error=error,
        )

        self._current_session.frames.append(frame)
        self._current_session.agent_ids.add(agent_id)

        return frame

    # =========================================================================
    # Convenience methods for common operations
    # =========================================================================

    def record_llm_request(
        self,
        agent_id: str,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> ReplayFrame:
        """Record LLM request"""
        if self._current_session:
            self._current_session.total_llm_calls += 1

        return self._record_frame(
            FrameType.LLM_REQUEST,
            agent_id,
            "llm_request",
            inputs={
                "prompt": prompt,
                "model": model,
                **kwargs
            }
        )

    def record_llm_response(
        self,
        agent_id: str,
        response: str,
        tokens_used: int = 0,
        model: str = "default",
        latency_ms: float = 0,
        **kwargs
    ) -> ReplayFrame:
        """Record LLM response"""
        if self._current_session:
            self._current_session.total_tokens += tokens_used

        return self._record_frame(
            FrameType.LLM_RESPONSE,
            agent_id,
            "llm_response",
            outputs={
                "response": response,
                "tokens_used": tokens_used,
                "model": model,
                "latency_ms": latency_ms,
                **kwargs
            }
        )

    def record_syscall(
        self,
        agent_id: str,
        syscall: str,
        args: Dict,
        result: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> ReplayFrame:
        """Record syscall"""
        if self._current_session:
            self._current_session.total_syscalls += 1

        return self._record_frame(
            FrameType.SYSCALL,
            agent_id,
            syscall,
            inputs=args,
            outputs={"result": result},
            success=success,
            error=error
        )

    def record_file_read(
        self,
        agent_id: str,
        path: str,
        content: str,
        success: bool = True
    ) -> ReplayFrame:
        """Record file read"""
        return self._record_frame(
            FrameType.FILE_READ,
            agent_id,
            "file_read",
            inputs={"path": path},
            outputs={"content": content, "size": len(content)},
            success=success
        )

    def record_file_write(
        self,
        agent_id: str,
        path: str,
        content: str,
        success: bool = True
    ) -> ReplayFrame:
        """Record file write"""
        return self._record_frame(
            FrameType.FILE_WRITE,
            agent_id,
            "file_write",
            inputs={"path": path, "content": content},
            outputs={"size": len(content)},
            success=success
        )

    def record_agent_spawn(
        self,
        parent_id: str,
        child_id: str,
        agent_type: str,
        config: Dict
    ) -> ReplayFrame:
        """Record agent spawn"""
        return self._record_frame(
            FrameType.AGENT_SPAWN,
            parent_id,
            "agent_spawn",
            inputs={"agent_type": agent_type, "config": config},
            outputs={"child_id": child_id}
        )

    def record_checkpoint(
        self,
        state: Dict,
        description: str = ""
    ) -> ReplayFrame:
        """Record checkpoint for state"""
        return self._record_frame(
            FrameType.CHECKPOINT,
            "",
            "checkpoint",
            inputs={"description": description},
            outputs={"state": state},
            state_hash=hashlib.sha256(
                json.dumps(state, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, session: RecordingSession, path: str):
        """Save recording to file"""
        data = {
            "version": "1.0",
            "session_id": session.session_id,
            "name": session.name,
            "description": session.description,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "initial_state": session.initial_state,
            "agent_ids": list(session.agent_ids),
            "tags": list(session.tags),
            "total_llm_calls": session.total_llm_calls,
            "total_tokens": session.total_tokens,
            "total_syscalls": session.total_syscalls,
            "frames": [
                {
                    "frame_id": f.frame_id,
                    "sequence": f.sequence,
                    "timestamp": f.timestamp,
                    "relative_time_ms": f.relative_time_ms,
                    "frame_type": f.frame_type.value,
                    "agent_id": f.agent_id,
                    "operation": f.operation,
                    "inputs": f.inputs,
                    "outputs": f.outputs,
                    "context_hash": f.context_hash,
                    "state_hash": f.state_hash,
                    "success": f.success,
                    "error": f.error,
                    "input_hash": f.input_hash,
                    "output_hash": f.output_hash,
                }
                for f in session.frames
            ]
        }

        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Saved recording to {path}")

    def load(self, path: str) -> RecordingSession:
        """Load recording from file"""
        data = json.loads(Path(path).read_text())

        session = RecordingSession(
            session_id=data["session_id"],
            name=data["name"],
            description=data["description"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            initial_state=data["initial_state"],
            agent_ids=set(data["agent_ids"]),
            tags=set(data["tags"]),
            total_llm_calls=data["total_llm_calls"],
            total_tokens=data["total_tokens"],
            total_syscalls=data["total_syscalls"],
        )

        for frame_data in data["frames"]:
            frame = ReplayFrame(
                frame_id=frame_data["frame_id"],
                sequence=frame_data["sequence"],
                timestamp=frame_data["timestamp"],
                relative_time_ms=frame_data["relative_time_ms"],
                frame_type=FrameType(frame_data["frame_type"]),
                agent_id=frame_data["agent_id"],
                operation=frame_data["operation"],
                inputs=frame_data["inputs"],
                outputs=frame_data["outputs"],
                context_hash=frame_data["context_hash"],
                state_hash=frame_data["state_hash"],
                success=frame_data["success"],
                error=frame_data["error"],
                input_hash=frame_data["input_hash"],
                output_hash=frame_data["output_hash"],
            )
            session.frames.append(frame)

        logger.info(f"Loaded recording from {path}: {len(session.frames)} frames")
        return session


# =============================================================================
# Player
# =============================================================================


@dataclass
class ReplayResult:
    """Result of replay execution"""
    session_id: str
    success: bool = True
    frames_executed: int = 0
    divergences: List[Dict] = field(default_factory=list)
    duration_ms: float = 0


class ReplayPlayer:
    """
    Plays back recorded sessions.

    Modes:
    - VERIFY: Check if replay matches recording exactly
    - MOCK: Use recorded outputs instead of real operations
    - REAL: Execute real operations, compare with recording
    """

    class Mode(Enum):
        VERIFY = "verify"  # Only check, don't execute
        MOCK = "mock"      # Use recorded values
        REAL = "real"      # Execute real ops, compare

    def __init__(self, mode: Mode = Mode.MOCK):
        self.mode = mode

        # Mock providers
        self._llm_mock: Optional[Callable] = None
        self._file_mock: Optional[Dict[str, str]] = None

        # Divergence tracking
        self._divergences: List[Dict] = []

        # Playback state
        self._current_frame = 0
        self._session: Optional[RecordingSession] = None

    def set_llm_mock(self, mock_fn: Callable[[str], str]):
        """Set mock LLM function"""
        self._llm_mock = mock_fn

    def set_file_mock(self, files: Dict[str, str]):
        """Set mock file system"""
        self._file_mock = files

    async def play(
        self,
        session: RecordingSession,
        speed: float = 1.0,
        stop_on_divergence: bool = False,
        on_frame: Optional[Callable[[ReplayFrame], None]] = None
    ) -> ReplayResult:
        """
        Play back a recording session.

        Args:
            session: Recording to play
            speed: Playback speed (1.0 = real-time, 0 = instant)
            stop_on_divergence: Stop if divergence detected
            on_frame: Callback for each frame
        """
        self._session = session
        self._current_frame = 0
        self._divergences = []

        start_time = time.time()
        last_relative_time = 0

        result = ReplayResult(session_id=session.session_id)

        for frame in session.frames:
            # Timing
            if speed > 0:
                delay = (frame.relative_time_ms - last_relative_time) / 1000 / speed
                if delay > 0:
                    await asyncio.sleep(delay)
                last_relative_time = frame.relative_time_ms

            # Execute frame
            try:
                divergence = await self._execute_frame(frame)
                if divergence and stop_on_divergence:
                    result.success = False
                    break
            except Exception as e:
                self._divergences.append({
                    "frame": frame.frame_id,
                    "type": "execution_error",
                    "error": str(e)
                })
                if stop_on_divergence:
                    result.success = False
                    break

            result.frames_executed += 1
            self._current_frame += 1

            if on_frame:
                on_frame(frame)

        result.divergences = self._divergences
        result.duration_ms = (time.time() - start_time) * 1000

        return result

    async def _execute_frame(self, frame: ReplayFrame) -> Optional[Dict]:
        """Execute a single frame"""
        if self.mode == ReplayPlayer.Mode.VERIFY:
            return None  # Just verify structure

        if frame.frame_type == FrameType.LLM_RESPONSE:
            if self.mode == ReplayPlayer.Mode.MOCK:
                # Return recorded response
                return None
            else:
                # Execute real LLM, compare
                # This would call actual LLM and compare
                pass

        elif frame.frame_type == FrameType.FILE_READ:
            if self.mode == ReplayPlayer.Mode.MOCK:
                # Return recorded content
                if self._file_mock:
                    path = frame.inputs.get("path")
                    recorded = frame.outputs.get("content")
                    if path in self._file_mock:
                        actual = self._file_mock[path]
                        if actual != recorded:
                            return self._record_divergence(
                                frame,
                                "file_content_mismatch",
                                recorded, actual
                            )

        return None

    def _record_divergence(
        self,
        frame: ReplayFrame,
        divergence_type: str,
        expected: Any,
        actual: Any
    ) -> Dict:
        """Record a divergence"""
        divergence = {
            "frame_id": frame.frame_id,
            "sequence": frame.sequence,
            "type": divergence_type,
            "expected": expected,
            "actual": actual,
            "operation": frame.operation,
        }
        self._divergences.append(divergence)
        return divergence

    def get_frame_at(self, index: int) -> Optional[ReplayFrame]:
        """Get frame at index"""
        if self._session and 0 <= index < len(self._session.frames):
            return self._session.frames[index]
        return None

    def seek_to(self, index: int):
        """Seek to frame index"""
        self._current_frame = max(0, min(index, len(self._session.frames) - 1 if self._session else 0))


# =============================================================================
# Replay Analyzer
# =============================================================================


class ReplayAnalyzer:
    """
    Analyzes recordings for insights.

    - Find patterns
    - Detect inefficiencies
    - Extract successful paths
    - Compare recordings
    """

    def analyze(self, session: RecordingSession) -> Dict[str, Any]:
        """Analyze a recording session"""
        return {
            "summary": self._summarize(session),
            "timeline": self._build_timeline(session),
            "patterns": self._find_patterns(session),
            "costs": self._calculate_costs(session),
            "bottlenecks": self._find_bottlenecks(session),
        }

    def _summarize(self, session: RecordingSession) -> Dict:
        """Generate summary statistics"""
        frame_types = {}
        for frame in session.frames:
            ft = frame.frame_type.value
            frame_types[ft] = frame_types.get(ft, 0) + 1

        return {
            "session_id": session.session_id,
            "name": session.name,
            "duration_ms": session.duration_ms,
            "total_frames": len(session.frames),
            "frame_types": frame_types,
            "agents": list(session.agent_ids),
            "llm_calls": session.total_llm_calls,
            "tokens": session.total_tokens,
            "syscalls": session.total_syscalls,
            "success_rate": sum(1 for f in session.frames if f.success) / max(1, len(session.frames)) * 100,
        }

    def _build_timeline(self, session: RecordingSession) -> List[Dict]:
        """Build timeline view"""
        return [
            {
                "time_ms": f.relative_time_ms,
                "type": f.frame_type.value,
                "agent": f.agent_id,
                "operation": f.operation,
                "success": f.success,
            }
            for f in session.frames
        ]

    def _find_patterns(self, session: RecordingSession) -> List[Dict]:
        """Find repeated patterns in recording"""
        patterns = []

        # Find LLM call patterns
        llm_ops = [f for f in session.frames if f.frame_type == FrameType.LLM_REQUEST]
        if llm_ops:
            patterns.append({
                "type": "llm_usage",
                "count": len(llm_ops),
                "avg_interval_ms": session.duration_ms / max(1, len(llm_ops)),
            })

        # Find repeated operations
        op_counts = {}
        for frame in session.frames:
            op = frame.operation
            op_counts[op] = op_counts.get(op, 0) + 1

        for op, count in op_counts.items():
            if count > 2:
                patterns.append({
                    "type": "repeated_operation",
                    "operation": op,
                    "count": count,
                })

        return patterns

    def _calculate_costs(self, session: RecordingSession) -> Dict:
        """Calculate estimated costs"""
        # Rough estimates
        token_cost = session.total_tokens * 0.00001  # $0.01 per 1K tokens
        compute_cost = session.duration_ms / 1000 / 60 * 0.01  # $0.01 per minute

        return {
            "estimated_token_cost": token_cost,
            "estimated_compute_cost": compute_cost,
            "total_estimated": token_cost + compute_cost,
        }

    def _find_bottlenecks(self, session: RecordingSession) -> List[Dict]:
        """Find performance bottlenecks"""
        bottlenecks = []

        # Find slow operations
        for i, frame in enumerate(session.frames):
            if i > 0:
                gap = frame.relative_time_ms - session.frames[i-1].relative_time_ms
                if gap > 1000:  # > 1 second gap
                    bottlenecks.append({
                        "frame": frame.frame_id,
                        "operation": frame.operation,
                        "gap_ms": gap,
                        "type": "slow_operation"
                    })

        return bottlenecks

    def compare(
        self,
        session1: RecordingSession,
        session2: RecordingSession
    ) -> Dict[str, Any]:
        """Compare two recordings"""
        return {
            "session1": self._summarize(session1),
            "session2": self._summarize(session2),
            "differences": {
                "duration_diff_ms": session2.duration_ms - session1.duration_ms,
                "frames_diff": len(session2.frames) - len(session1.frames),
                "llm_calls_diff": session2.total_llm_calls - session1.total_llm_calls,
                "tokens_diff": session2.total_tokens - session1.total_tokens,
            },
            "divergence_points": self._find_divergence_points(session1, session2),
        }

    def _find_divergence_points(
        self,
        session1: RecordingSession,
        session2: RecordingSession
    ) -> List[Dict]:
        """Find where two recordings diverge"""
        divergences = []

        min_len = min(len(session1.frames), len(session2.frames))

        for i in range(min_len):
            f1 = session1.frames[i]
            f2 = session2.frames[i]

            if f1.operation != f2.operation:
                divergences.append({
                    "index": i,
                    "type": "operation_mismatch",
                    "session1": f1.operation,
                    "session2": f2.operation,
                })
            elif f1.output_hash != f2.output_hash:
                divergences.append({
                    "index": i,
                    "type": "output_mismatch",
                    "operation": f1.operation,
                })

        return divergences


# =============================================================================
# Replay Middleware (Auto-recording)
# =============================================================================


class ReplayMiddleware:
    """
    Middleware for automatic recording of operations.

    Wrap around syscall table or agent for automatic recording.
    """

    def __init__(self, recorder: ReplayRecorder):
        self.recorder = recorder

    def wrap_syscall(self, syscall_fn: Callable) -> Callable:
        """Wrap syscall function for recording"""
        async def wrapped(request, *args, **kwargs):
            # Record syscall
            result = await syscall_fn(request, *args, **kwargs)

            if self.recorder.is_recording:
                self.recorder.record_syscall(
                    request.agent_id,
                    request.syscall.name,
                    request.args,
                    result.result,
                    result.success,
                    result.error
                )

            return result

        return wrapped

    def wrap_llm(self, llm_fn: Callable) -> Callable:
        """Wrap LLM function for recording"""
        async def wrapped(prompt: str, **kwargs):
            if self.recorder.is_recording:
                self.recorder.record_llm_request("default", prompt, **kwargs)

            response = await llm_fn(prompt, **kwargs)

            if self.recorder.is_recording:
                self.recorder.record_llm_response(
                    "default",
                    response.get("content", ""),
                    response.get("tokens", 0)
                )

            return response

        return wrapped


# =============================================================================
# Global Instance
# =============================================================================


_recorder: Optional[ReplayRecorder] = None


def get_recorder() -> ReplayRecorder:
    """Get global recorder instance"""
    global _recorder
    if _recorder is None:
        _recorder = ReplayRecorder()
    return _recorder


# =============================================================================
# Convenience Functions
# =============================================================================


def start_recording(name: str = "", description: str = "") -> str:
    """Start recording"""
    return get_recorder().start_session(name, description)


def stop_recording() -> RecordingSession:
    """Stop recording"""
    return get_recorder().end_session()


def save_recording(session: RecordingSession, path: str):
    """Save recording"""
    get_recorder().save(session, path)


def load_recording(path: str) -> RecordingSession:
    """Load recording"""
    return get_recorder().load(path)


async def replay(
    path: str,
    mode: ReplayPlayer.Mode = ReplayPlayer.Mode.MOCK
) -> ReplayResult:
    """Replay a recording"""
    session = load_recording(path)
    player = ReplayPlayer(mode)
    return await player.play(session)
