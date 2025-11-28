# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX FlowIntegrity - CET-inspired Control Flow Protection for AI Agents

Provides runtime protection against unauthorized control flow transitions:
- Shadow execution trace validation
- Allowed transition graph enforcement
- Anomaly detection and response
- Hard fail / soft rollback policies

Inspired by Intel CET (Control-flow Enforcement Technology):
- Shadow stack for return address protection → Shadow trace for state transitions
- Indirect branch tracking → Allowed transition graph
- Hardware enforcement → Runtime verification

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  FlowIntegrity Engine                                       │
    │  ├─ TransitionGraph: Defines allowed state transitions      │
    │  ├─ ShadowTrace: Records expected execution path            │
    │  ├─ Verifier: Validates actual vs expected transitions      │
    │  └─ Enforcer: Applies policy on violations                  │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("bbx.flow_integrity")


# =============================================================================
# Enums and Types
# =============================================================================

class FlowState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    TOOL_CALLING = "tool_calling"
    WAITING = "waiting"
    PROCESSING = "processing"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class TransitionType(Enum):
    """Types of state transitions"""
    NORMAL = auto()      # Expected transition
    CONDITIONAL = auto() # Depends on condition
    EMERGENCY = auto()   # Emergency/error handling
    ROLLBACK = auto()    # State rollback


class ViolationType(Enum):
    """Types of control flow violations"""
    INVALID_TRANSITION = "invalid_transition"
    UNEXPECTED_STATE = "unexpected_state"
    TRACE_MISMATCH = "trace_mismatch"
    LOOP_DETECTED = "loop_detected"
    TIMEOUT = "timeout"
    DEPTH_EXCEEDED = "depth_exceeded"
    POLICY_VIOLATION = "policy_violation"


class EnforcementAction(Enum):
    """Actions to take on violation"""
    ALLOW = "allow"           # Log and continue
    WARN = "warn"             # Warn and continue
    SOFT_BLOCK = "soft_block" # Block with fallback
    HARD_BLOCK = "hard_block" # Block immediately
    ROLLBACK = "rollback"     # Rollback to last safe state
    TERMINATE = "terminate"   # Terminate agent


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Transition:
    """Defines an allowed state transition"""
    from_state: FlowState
    to_state: FlowState
    transition_type: TransitionType = TransitionType.NORMAL
    condition: Optional[Callable[[Dict], bool]] = None
    description: str = ""
    max_frequency: Optional[int] = None  # Max times per minute
    requires_auth: bool = False


@dataclass
class TraceEntry:
    """Entry in the shadow execution trace"""
    state: FlowState
    timestamp: datetime
    agent_id: str
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = f"{self.state.value}:{self.agent_id}:{self.workflow_id}:{self.step_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class Violation:
    """Represents a control flow violation"""
    violation_type: ViolationType
    timestamp: datetime
    agent_id: str
    expected_states: List[FlowState]
    actual_state: FlowState
    context: Dict[str, Any] = field(default_factory=dict)
    severity: int = 1  # 1-5, 5 is most severe


@dataclass
class FlowPolicy:
    """Policy for handling violations"""
    name: str
    violation_types: List[ViolationType]
    action: EnforcementAction
    max_violations: int = 3  # Before escalation
    escalation_action: Optional[EnforcementAction] = None
    cooldown_sec: float = 60.0


@dataclass
class FlowIntegrityConfig:
    """Configuration for FlowIntegrity"""
    enabled: bool = True
    max_trace_depth: int = 1000
    max_loop_count: int = 10
    trace_retention_sec: float = 3600.0
    strict_mode: bool = False  # Hard fail on any violation
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 0.8


# =============================================================================
# Transition Graph
# =============================================================================

class TransitionGraph:
    """
    Defines the allowed state transition graph for agents.

    Like CET's IBT (Indirect Branch Tracking), this validates
    that agents only make authorized state transitions.
    """

    def __init__(self):
        self._transitions: Dict[FlowState, List[Transition]] = defaultdict(list)
        self._all_states: Set[FlowState] = set()
        self._setup_default_transitions()

    def _setup_default_transitions(self):
        """Setup default allowed transitions for agents"""
        default_transitions = [
            # Initialization
            Transition(FlowState.IDLE, FlowState.INITIALIZING),
            Transition(FlowState.INITIALIZING, FlowState.PLANNING),
            Transition(FlowState.INITIALIZING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Planning
            Transition(FlowState.PLANNING, FlowState.EXECUTING),
            Transition(FlowState.PLANNING, FlowState.WAITING),
            Transition(FlowState.PLANNING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Execution
            Transition(FlowState.EXECUTING, FlowState.TOOL_CALLING),
            Transition(FlowState.EXECUTING, FlowState.PROCESSING),
            Transition(FlowState.EXECUTING, FlowState.COMMITTING),
            Transition(FlowState.EXECUTING, FlowState.PLANNING),  # Re-planning
            Transition(FlowState.EXECUTING, FlowState.FAILED, TransitionType.EMERGENCY),
            Transition(FlowState.EXECUTING, FlowState.ROLLING_BACK, TransitionType.ROLLBACK),

            # Tool calling
            Transition(FlowState.TOOL_CALLING, FlowState.PROCESSING),
            Transition(FlowState.TOOL_CALLING, FlowState.EXECUTING),
            Transition(FlowState.TOOL_CALLING, FlowState.WAITING),
            Transition(FlowState.TOOL_CALLING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Processing
            Transition(FlowState.PROCESSING, FlowState.EXECUTING),
            Transition(FlowState.PROCESSING, FlowState.COMMITTING),
            Transition(FlowState.PROCESSING, FlowState.PLANNING),
            Transition(FlowState.PROCESSING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Waiting
            Transition(FlowState.WAITING, FlowState.EXECUTING),
            Transition(FlowState.WAITING, FlowState.PLANNING),
            Transition(FlowState.WAITING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Committing
            Transition(FlowState.COMMITTING, FlowState.COMPLETED),
            Transition(FlowState.COMMITTING, FlowState.EXECUTING),  # More work
            Transition(FlowState.COMMITTING, FlowState.ROLLING_BACK, TransitionType.ROLLBACK),
            Transition(FlowState.COMMITTING, FlowState.FAILED, TransitionType.EMERGENCY),

            # Rolling back
            Transition(FlowState.ROLLING_BACK, FlowState.EXECUTING),
            Transition(FlowState.ROLLING_BACK, FlowState.PLANNING),
            Transition(FlowState.ROLLING_BACK, FlowState.IDLE),
            Transition(FlowState.ROLLING_BACK, FlowState.FAILED, TransitionType.EMERGENCY),

            # Terminal states can restart
            Transition(FlowState.COMPLETED, FlowState.IDLE),
            Transition(FlowState.FAILED, FlowState.IDLE),
            Transition(FlowState.FAILED, FlowState.ROLLING_BACK, TransitionType.ROLLBACK),

            # Suspension
            Transition(FlowState.EXECUTING, FlowState.SUSPENDED),
            Transition(FlowState.PLANNING, FlowState.SUSPENDED),
            Transition(FlowState.SUSPENDED, FlowState.EXECUTING),
            Transition(FlowState.SUSPENDED, FlowState.PLANNING),
            Transition(FlowState.SUSPENDED, FlowState.IDLE),
        ]

        for t in default_transitions:
            self.add_transition(t)

    def add_transition(self, transition: Transition):
        """Add an allowed transition"""
        self._transitions[transition.from_state].append(transition)
        self._all_states.add(transition.from_state)
        self._all_states.add(transition.to_state)

    def remove_transition(self, from_state: FlowState, to_state: FlowState):
        """Remove a transition"""
        self._transitions[from_state] = [
            t for t in self._transitions[from_state]
            if t.to_state != to_state
        ]

    def is_valid_transition(
        self,
        from_state: FlowState,
        to_state: FlowState,
        context: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Transition]]:
        """Check if a transition is valid"""
        for transition in self._transitions[from_state]:
            if transition.to_state == to_state:
                # Check condition if present
                if transition.condition:
                    if not transition.condition(context or {}):
                        continue
                return True, transition
        return False, None

    def get_allowed_transitions(self, from_state: FlowState) -> List[FlowState]:
        """Get all allowed target states from a given state"""
        return [t.to_state for t in self._transitions[from_state]]

    def get_all_states(self) -> Set[FlowState]:
        """Get all states in the graph"""
        return self._all_states.copy()

    def to_dict(self) -> Dict:
        """Export graph as dictionary"""
        result = {}
        for from_state, transitions in self._transitions.items():
            result[from_state.value] = [
                {
                    "to": t.to_state.value,
                    "type": t.transition_type.name,
                    "description": t.description,
                }
                for t in transitions
            ]
        return result


# =============================================================================
# Shadow Trace
# =============================================================================

class ShadowTrace:
    """
    Shadow execution trace for agents.

    Like CET's shadow stack protects return addresses,
    ShadowTrace protects the expected execution path.
    """

    def __init__(self, config: FlowIntegrityConfig):
        self.config = config
        self._traces: Dict[str, List[TraceEntry]] = defaultdict(list)
        self._expected_next: Dict[str, List[FlowState]] = {}

    def push(self, entry: TraceEntry):
        """Push a trace entry"""
        agent_id = entry.agent_id
        self._traces[agent_id].append(entry)

        # Trim if too deep
        if len(self._traces[agent_id]) > self.config.max_trace_depth:
            self._traces[agent_id] = self._traces[agent_id][-self.config.max_trace_depth:]

    def pop(self, agent_id: str) -> Optional[TraceEntry]:
        """Pop the last trace entry"""
        if self._traces[agent_id]:
            return self._traces[agent_id].pop()
        return None

    def peek(self, agent_id: str) -> Optional[TraceEntry]:
        """Peek at the current trace entry"""
        if self._traces[agent_id]:
            return self._traces[agent_id][-1]
        return None

    def get_trace(self, agent_id: str, limit: int = 100) -> List[TraceEntry]:
        """Get trace for an agent"""
        return self._traces[agent_id][-limit:]

    def set_expected_next(self, agent_id: str, states: List[FlowState]):
        """Set expected next states"""
        self._expected_next[agent_id] = states

    def get_expected_next(self, agent_id: str) -> List[FlowState]:
        """Get expected next states"""
        return self._expected_next.get(agent_id, [])

    def verify_state(self, agent_id: str, state: FlowState) -> bool:
        """Verify state matches expected"""
        expected = self._expected_next.get(agent_id)
        if expected is None:
            return True  # No expectation set
        return state in expected

    def detect_loop(self, agent_id: str, window: int = 20) -> bool:
        """Detect if agent is stuck in a loop"""
        trace = self._traces[agent_id][-window:]
        if len(trace) < self.config.max_loop_count:
            return False

        # Check for repeated patterns
        states = [e.state for e in trace]
        for pattern_len in range(2, window // 2):
            pattern = states[-pattern_len:]
            matches = 0
            for i in range(len(states) - pattern_len):
                if states[i:i + pattern_len] == pattern:
                    matches += 1
            if matches >= self.config.max_loop_count:
                return True

        return False

    def clear(self, agent_id: str):
        """Clear trace for an agent"""
        self._traces[agent_id] = []
        self._expected_next.pop(agent_id, None)

    def get_trace_hash(self, agent_id: str) -> str:
        """Get hash of the trace for verification"""
        trace = self._traces[agent_id]
        data = ":".join(e.hash for e in trace[-10:])
        return hashlib.sha256(data.encode()).hexdigest()[:32]


# =============================================================================
# Flow Integrity Engine
# =============================================================================

class FlowIntegrityEngine:
    """
    Main FlowIntegrity engine - CET for AI Agents.

    Provides:
    - Control flow validation
    - Shadow trace management
    - Violation detection
    - Policy enforcement
    """

    def __init__(self, config: Optional[FlowIntegrityConfig] = None):
        self.config = config or FlowIntegrityConfig()
        self.graph = TransitionGraph()
        self.shadow_trace = ShadowTrace(self.config)
        self._policies: List[FlowPolicy] = []
        self._violation_counts: Dict[str, Dict[ViolationType, int]] = defaultdict(lambda: defaultdict(int))
        self._agent_states: Dict[str, FlowState] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Setup default enforcement policies"""
        self._policies = [
            FlowPolicy(
                name="invalid_transition_policy",
                violation_types=[ViolationType.INVALID_TRANSITION],
                action=EnforcementAction.SOFT_BLOCK,
                max_violations=3,
                escalation_action=EnforcementAction.ROLLBACK,
            ),
            FlowPolicy(
                name="loop_detection_policy",
                violation_types=[ViolationType.LOOP_DETECTED],
                action=EnforcementAction.WARN,
                max_violations=2,
                escalation_action=EnforcementAction.TERMINATE,
            ),
            FlowPolicy(
                name="trace_mismatch_policy",
                violation_types=[ViolationType.TRACE_MISMATCH],
                action=EnforcementAction.ROLLBACK,
            ),
            FlowPolicy(
                name="depth_exceeded_policy",
                violation_types=[ViolationType.DEPTH_EXCEEDED],
                action=EnforcementAction.HARD_BLOCK,
            ),
        ]

    def add_policy(self, policy: FlowPolicy):
        """Add an enforcement policy"""
        self._policies.append(policy)

    def register_callback(self, event: str, callback: Callable):
        """Register callback for events (violation, transition, etc.)"""
        self._callbacks[event].append(callback)

    async def _emit_event(self, event: str, data: Any):
        """Emit an event to callbacks"""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_state(self, agent_id: str) -> FlowState:
        """Get current state of an agent"""
        return self._agent_states.get(agent_id, FlowState.IDLE)

    async def transition(
        self,
        agent_id: str,
        to_state: FlowState,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[Violation]]:
        """
        Attempt a state transition.

        Returns (success, violation) tuple.
        """
        if not self.config.enabled:
            self._agent_states[agent_id] = to_state
            return True, None

        from_state = self.get_state(agent_id)

        # Check if transition is valid
        is_valid, transition = self.graph.is_valid_transition(
            from_state, to_state, context
        )

        if not is_valid and not force:
            # Create violation
            violation = Violation(
                violation_type=ViolationType.INVALID_TRANSITION,
                timestamp=datetime.now(),
                agent_id=agent_id,
                expected_states=self.graph.get_allowed_transitions(from_state),
                actual_state=to_state,
                context=context or {},
                severity=3,
            )

            # Apply policy
            action = await self._apply_policy(violation)

            if action in [EnforcementAction.HARD_BLOCK, EnforcementAction.TERMINATE]:
                return False, violation

            if action == EnforcementAction.ROLLBACK:
                # Rollback to last safe state
                await self._rollback_to_safe_state(agent_id)
                return False, violation

        # Check shadow trace expectation
        if not self.shadow_trace.verify_state(agent_id, to_state) and not force:
            violation = Violation(
                violation_type=ViolationType.TRACE_MISMATCH,
                timestamp=datetime.now(),
                agent_id=agent_id,
                expected_states=self.shadow_trace.get_expected_next(agent_id),
                actual_state=to_state,
                severity=2,
            )

            action = await self._apply_policy(violation)
            if action == EnforcementAction.HARD_BLOCK:
                return False, violation

        # Check for loops
        if self.shadow_trace.detect_loop(agent_id):
            violation = Violation(
                violation_type=ViolationType.LOOP_DETECTED,
                timestamp=datetime.now(),
                agent_id=agent_id,
                expected_states=[],
                actual_state=to_state,
                severity=4,
            )

            action = await self._apply_policy(violation)
            if action in [EnforcementAction.HARD_BLOCK, EnforcementAction.TERMINATE]:
                return False, violation

        # Record transition
        entry = TraceEntry(
            state=to_state,
            timestamp=datetime.now(),
            agent_id=agent_id,
            workflow_id=workflow_id,
            step_id=step_id,
            metadata=context or {},
        )
        self.shadow_trace.push(entry)
        self._agent_states[agent_id] = to_state

        # Set expected next states
        allowed_next = self.graph.get_allowed_transitions(to_state)
        self.shadow_trace.set_expected_next(agent_id, allowed_next)

        # Emit transition event
        await self._emit_event("transition", {
            "agent_id": agent_id,
            "from_state": from_state,
            "to_state": to_state,
            "workflow_id": workflow_id,
            "step_id": step_id,
        })

        return True, None

    async def _apply_policy(self, violation: Violation) -> EnforcementAction:
        """Apply policy based on violation"""
        agent_id = violation.agent_id
        v_type = violation.violation_type

        # Increment violation count
        self._violation_counts[agent_id][v_type] += 1
        count = self._violation_counts[agent_id][v_type]

        # Emit violation event
        await self._emit_event("violation", violation)

        # Find matching policy
        for policy in self._policies:
            if v_type in policy.violation_types:
                if count >= policy.max_violations and policy.escalation_action:
                    logger.warning(
                        f"FlowIntegrity escalation: {agent_id} - {v_type.value} "
                        f"(count={count}) -> {policy.escalation_action.value}"
                    )
                    return policy.escalation_action

                logger.warning(
                    f"FlowIntegrity violation: {agent_id} - {v_type.value} "
                    f"-> {policy.action.value}"
                )
                return policy.action

        # Default action
        if self.config.strict_mode:
            return EnforcementAction.HARD_BLOCK
        return EnforcementAction.WARN

    async def _rollback_to_safe_state(self, agent_id: str):
        """Rollback agent to last safe state"""
        trace = self.shadow_trace.get_trace(agent_id, limit=10)

        # Find last safe state (IDLE, PLANNING, EXECUTING)
        safe_states = {FlowState.IDLE, FlowState.PLANNING, FlowState.EXECUTING}
        for entry in reversed(trace):
            if entry.state in safe_states:
                self._agent_states[agent_id] = entry.state
                logger.info(f"Rolled back {agent_id} to {entry.state.value}")
                return

        # Default to IDLE
        self._agent_states[agent_id] = FlowState.IDLE

    def reset_violations(self, agent_id: str):
        """Reset violation counts for an agent"""
        self._violation_counts[agent_id] = defaultdict(int)

    def get_violations(self, agent_id: str) -> Dict[ViolationType, int]:
        """Get violation counts for an agent"""
        return dict(self._violation_counts[agent_id])

    def get_trace(self, agent_id: str, limit: int = 100) -> List[TraceEntry]:
        """Get execution trace for an agent"""
        return self.shadow_trace.get_trace(agent_id, limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get FlowIntegrity statistics"""
        total_violations = sum(
            sum(counts.values())
            for counts in self._violation_counts.values()
        )

        return {
            "enabled": self.config.enabled,
            "strict_mode": self.config.strict_mode,
            "active_agents": len(self._agent_states),
            "total_violations": total_violations,
            "violations_by_type": {
                v_type.value: sum(
                    counts.get(v_type, 0)
                    for counts in self._violation_counts.values()
                )
                for v_type in ViolationType
            },
            "policies_count": len(self._policies),
        }


# =============================================================================
# Global Instance
# =============================================================================

_flow_integrity: Optional[FlowIntegrityEngine] = None


def get_flow_integrity() -> FlowIntegrityEngine:
    """Get global FlowIntegrity instance"""
    global _flow_integrity
    if _flow_integrity is None:
        _flow_integrity = FlowIntegrityEngine()
    return _flow_integrity
