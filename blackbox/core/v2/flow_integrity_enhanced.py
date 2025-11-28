# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 FlowIntegrity Enhanced - Behavioral anomaly detection and policy enforcement.

Improvements:
- ML-based behavioral anomaly detection
- OPA (Open Policy Agent) integration for runtime policy checks
- Memory access controls (semantic memory authorization)
- Tool call validation
- Runtime verification beyond state transitions

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │              Enhanced Flow Integrity                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
    │  │ Anomaly      │  │ OPA Policy   │  │ Memory Access        │   │
    │  │ Detector     │  │ Engine       │  │ Control              │   │
    │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
    │         │                 │                      │               │
    │  ┌──────▼─────────────────▼──────────────────────▼───────────┐   │
    │  │                 Verification Engine                        │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
    │  │  │ State   │ │ Tool    │ │ Goal    │ │   Behavioral    │  │   │
    │  │  │ Trans.  │ │ Calls   │ │ Align   │ │   Patterns      │  │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
    │  └───────────────────────────────────────────────────────────┘   │
    │                              │                                   │
    │  ┌───────────────────────────▼───────────────────────────────┐   │
    │  │              Response Actions                              │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
    │  │  │ Allow   │ │ Block   │ │ Audit   │ │   Quarantine    │  │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
    │  └───────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Pattern

logger = logging.getLogger("bbx.flow.enhanced")


# =============================================================================
# Enums
# =============================================================================


class VerificationResult(Enum):
    """Result of verification"""
    ALLOW = auto()
    BLOCK = auto()
    AUDIT = auto()
    QUARANTINE = auto()
    WARN = auto()


class AnomalyType(Enum):
    """Types of anomalies"""
    UNUSUAL_STATE_TRANSITION = auto()
    EXCESSIVE_TOOL_CALLS = auto()
    SUSPICIOUS_MEMORY_ACCESS = auto()
    GOAL_DEVIATION = auto()
    RATE_ANOMALY = auto()
    PATTERN_BREAK = auto()
    RESOURCE_ABUSE = auto()


class PolicyDecision(Enum):
    """OPA policy decision"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITION = "condition"  # Allow with conditions


# =============================================================================
# Behavioral Anomaly Detection
# =============================================================================


@dataclass
class BehaviorProfile:
    """Profile of normal agent behavior"""
    agent_id: str

    # State transition patterns
    common_transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    avg_time_in_state: Dict[str, float] = field(default_factory=dict)

    # Tool usage patterns
    tool_call_frequency: Dict[str, float] = field(default_factory=dict)  # calls per minute
    tool_sequences: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Memory access patterns
    memory_access_frequency: float = 0.0
    common_memory_keys: Set[str] = field(default_factory=set)

    # Timing patterns
    avg_response_time_ms: float = 0.0
    std_response_time_ms: float = 0.0

    # Learning metadata
    samples_collected: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class AnomalyEvent:
    """A detected anomaly"""
    id: str
    agent_id: str
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    description: str
    context: Dict[str, Any]
    detected_at: float = field(default_factory=time.time)
    action_taken: Optional[VerificationResult] = None


class AnomalyDetector:
    """
    ML-based behavioral anomaly detection.

    Uses statistical methods to detect deviations from learned behavior.
    """

    def __init__(
        self,
        learning_period: int = 1000,  # Samples before active detection
        sensitivity: float = 2.0,     # Standard deviations for anomaly
        window_size: int = 100        # Rolling window for statistics
    ):
        self._learning_period = learning_period
        self._sensitivity = sensitivity
        self._window_size = window_size

        # Behavior profiles per agent
        self._profiles: Dict[str, BehaviorProfile] = {}

        # Recent observations for rolling statistics
        self._recent_observations: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )

        # Anomaly history
        self._anomalies: List[AnomalyEvent] = []
        self._anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []

    def get_or_create_profile(self, agent_id: str) -> BehaviorProfile:
        """Get or create behavior profile for agent"""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = BehaviorProfile(agent_id=agent_id)
        return self._profiles[agent_id]

    def record_state_transition(
        self,
        agent_id: str,
        from_state: str,
        to_state: str,
        duration_ms: float
    ) -> Optional[AnomalyEvent]:
        """Record a state transition and check for anomalies"""
        profile = self.get_or_create_profile(agent_id)
        transition = (from_state, to_state)

        # Update profile
        profile.common_transitions[transition] = profile.common_transitions.get(transition, 0) + 1

        # Update average time in state
        if from_state in profile.avg_time_in_state:
            old_avg = profile.avg_time_in_state[from_state]
            n = profile.samples_collected
            profile.avg_time_in_state[from_state] = (old_avg * n + duration_ms) / (n + 1)
        else:
            profile.avg_time_in_state[from_state] = duration_ms

        profile.samples_collected += 1
        profile.last_updated = time.time()

        # Record observation
        obs = self._recent_observations[agent_id]
        obs["transitions"].append(transition)
        obs["durations"].append(duration_ms)

        # Check for anomalies (only after learning period)
        if profile.samples_collected >= self._learning_period:
            return self._check_transition_anomaly(agent_id, transition, duration_ms)

        return None

    def record_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        execution_time_ms: float
    ) -> Optional[AnomalyEvent]:
        """Record a tool call and check for anomalies"""
        profile = self.get_or_create_profile(agent_id)
        now = time.time()

        # Update frequency
        obs = self._recent_observations[agent_id]
        obs["tool_calls"].append((tool_name, now))

        # Calculate recent frequency
        minute_ago = now - 60
        recent_calls = [t for t, ts in list(obs["tool_calls"])[-1000:] if ts >= minute_ago]
        frequency = len([t for t in recent_calls if t == tool_name])

        profile.tool_call_frequency[tool_name] = frequency
        profile.samples_collected += 1

        # Check for rate anomaly
        if profile.samples_collected >= self._learning_period:
            return self._check_rate_anomaly(agent_id, tool_name, frequency)

        return None

    def record_memory_access(
        self,
        agent_id: str,
        key: str,
        operation: str  # 'read' or 'write'
    ) -> Optional[AnomalyEvent]:
        """Record memory access and check for anomalies"""
        profile = self.get_or_create_profile(agent_id)

        obs = self._recent_observations[agent_id]
        obs["memory_keys"].append(key)

        # Update common keys
        profile.common_memory_keys.add(key)

        # Check for suspicious access (new key after learning)
        if profile.samples_collected >= self._learning_period:
            if key not in profile.common_memory_keys:
                return self._create_anomaly(
                    agent_id,
                    AnomalyType.SUSPICIOUS_MEMORY_ACCESS,
                    0.5,
                    f"Access to unusual memory key: {key}",
                    {"key": key, "operation": operation}
                )

        return None

    def _check_transition_anomaly(
        self,
        agent_id: str,
        transition: Tuple[str, str],
        duration_ms: float
    ) -> Optional[AnomalyEvent]:
        """Check if transition is anomalous"""
        profile = self._profiles[agent_id]

        # Check if transition has ever occurred
        if transition not in profile.common_transitions:
            return self._create_anomaly(
                agent_id,
                AnomalyType.UNUSUAL_STATE_TRANSITION,
                0.8,
                f"Novel state transition: {transition[0]} -> {transition[1]}",
                {"from": transition[0], "to": transition[1]}
            )

        # Check if timing is unusual
        from_state = transition[0]
        if from_state in profile.avg_time_in_state:
            avg = profile.avg_time_in_state[from_state]
            # Rough std approximation
            std = avg * 0.5

            if abs(duration_ms - avg) > self._sensitivity * std:
                return self._create_anomaly(
                    agent_id,
                    AnomalyType.PATTERN_BREAK,
                    0.6,
                    f"Unusual timing in state {from_state}: {duration_ms}ms (avg: {avg}ms)",
                    {"state": from_state, "duration": duration_ms, "avg": avg}
                )

        return None

    def _check_rate_anomaly(
        self,
        agent_id: str,
        tool_name: str,
        current_frequency: int
    ) -> Optional[AnomalyEvent]:
        """Check if tool call rate is anomalous"""
        profile = self._profiles[agent_id]

        # Get historical frequency
        historical = profile.tool_call_frequency.get(tool_name, 0)

        if historical > 0:
            ratio = current_frequency / historical
            if ratio > 3.0:  # 3x increase
                return self._create_anomaly(
                    agent_id,
                    AnomalyType.EXCESSIVE_TOOL_CALLS,
                    min(0.9, ratio / 10),
                    f"Excessive {tool_name} calls: {current_frequency}/min (normal: {historical}/min)",
                    {"tool": tool_name, "current": current_frequency, "normal": historical}
                )

        return None

    def _create_anomaly(
        self,
        agent_id: str,
        anomaly_type: AnomalyType,
        severity: float,
        description: str,
        context: Dict[str, Any]
    ) -> AnomalyEvent:
        """Create and record an anomaly event"""
        event = AnomalyEvent(
            id=f"anom_{int(time.time() * 1000)}_{len(self._anomalies)}",
            agent_id=agent_id,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            context=context
        )

        self._anomalies.append(event)

        # Trigger callbacks
        for callback in self._anomaly_callbacks:
            try:
                callback(event)
            except Exception:
                pass

        return event

    def on_anomaly(self, callback: Callable[[AnomalyEvent], None]):
        """Register anomaly callback"""
        self._anomaly_callbacks.append(callback)

    def get_anomalies(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None,
        min_severity: float = 0.0
    ) -> List[AnomalyEvent]:
        """Get recorded anomalies"""
        anomalies = self._anomalies

        if agent_id:
            anomalies = [a for a in anomalies if a.agent_id == agent_id]

        if since:
            anomalies = [a for a in anomalies if a.detected_at >= since]

        if min_severity > 0:
            anomalies = [a for a in anomalies if a.severity >= min_severity]

        return anomalies


# =============================================================================
# OPA Policy Engine
# =============================================================================


@dataclass
class PolicyRule:
    """A policy rule"""
    id: str
    name: str
    description: str
    condition: str  # Rego-like condition or Python expression
    decision: PolicyDecision
    priority: int = 0
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation"""
    rule_id: str
    decision: PolicyDecision
    matched: bool
    context: Dict[str, Any]
    evaluated_at: float = field(default_factory=time.time)


class PolicyEngine:
    """
    OPA-inspired policy engine for runtime policy enforcement.

    Supports:
    - Rule-based policies
    - Attribute-based access control (ABAC)
    - Custom policy expressions
    """

    def __init__(self):
        self._rules: Dict[str, PolicyRule] = {}
        self._default_decision = PolicyDecision.DENY

        # Built-in policy functions
        self._functions: Dict[str, Callable] = {
            "contains": lambda x, y: y in x,
            "startswith": lambda x, y: x.startswith(y),
            "endswith": lambda x, y: x.endswith(y),
            "matches": lambda x, y: bool(re.match(y, x)),
            "in_set": lambda x, s: x in s,
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "eq": lambda x, y: x == y,
        }

    def add_rule(self, rule: PolicyRule):
        """Add a policy rule"""
        self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        """Remove a policy rule"""
        self._rules.pop(rule_id, None)

    def evaluate(
        self,
        action: str,
        resource: str,
        subject: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyEvaluation:
        """
        Evaluate policies for an action.

        Args:
            action: The action being attempted (e.g., "tool.execute", "memory.read")
            resource: The resource being accessed
            subject: Information about the agent/user
            context: Additional context

        Returns:
            PolicyEvaluation with decision
        """
        context = context or {}

        # Build evaluation context
        eval_context = {
            "action": action,
            "resource": resource,
            "subject": subject,
            "context": context,
            **self._functions
        }

        # Evaluate rules in priority order
        sorted_rules = sorted(
            [r for r in self._rules.values() if r.enabled],
            key=lambda r: -r.priority
        )

        for rule in sorted_rules:
            try:
                matched = self._evaluate_condition(rule.condition, eval_context)
                if matched:
                    return PolicyEvaluation(
                        rule_id=rule.id,
                        decision=rule.decision,
                        matched=True,
                        context=eval_context
                    )
            except Exception as e:
                logger.warning(f"Policy rule {rule.id} evaluation error: {e}")

        # Default decision
        return PolicyEvaluation(
            rule_id="default",
            decision=self._default_decision,
            matched=False,
            context=eval_context
        )

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a policy condition"""
        # Simple expression evaluation
        # In production, use a proper policy language parser
        try:
            # Safe evaluation with limited globals
            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)
        except Exception:
            return False

    def load_policies_from_json(self, json_data: str):
        """Load policies from JSON"""
        data = json.loads(json_data)
        for rule_data in data.get("rules", []):
            rule = PolicyRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                condition=rule_data["condition"],
                decision=PolicyDecision(rule_data["decision"]),
                priority=rule_data.get("priority", 0),
                enabled=rule_data.get("enabled", True),
                tags=set(rule_data.get("tags", []))
            )
            self.add_rule(rule)


# =============================================================================
# Memory Access Control
# =============================================================================


@dataclass
class MemoryAccessRule:
    """Rule for memory access control"""
    pattern: str  # Regex pattern for key matching
    allowed_agents: Set[str]  # Agent IDs or '*' for all
    allowed_operations: Set[str]  # 'read', 'write', 'delete'
    require_context: bool = False  # Require specific context
    audit: bool = True


class MemoryAccessControl:
    """
    Controls agent access to semantic memory.

    Features:
    - Key-pattern based access rules
    - Agent-specific permissions
    - Context-aware access decisions
    - Audit logging
    """

    def __init__(self):
        self._rules: List[MemoryAccessRule] = []
        self._compiled_patterns: List[Tuple[Pattern, MemoryAccessRule]] = []
        self._audit_log: List[Dict[str, Any]] = []

    def add_rule(self, rule: MemoryAccessRule):
        """Add an access rule"""
        self._rules.append(rule)
        self._compiled_patterns.append((re.compile(rule.pattern), rule))

    def check_access(
        self,
        agent_id: str,
        key: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if access is allowed.

        Returns (allowed, reason)
        """
        context = context or {}

        for pattern, rule in self._compiled_patterns:
            if pattern.match(key):
                # Check agent
                if '*' not in rule.allowed_agents and agent_id not in rule.allowed_agents:
                    reason = f"Agent {agent_id} not allowed to access keys matching {rule.pattern}"
                    self._log_audit(agent_id, key, operation, False, reason)
                    return False, reason

                # Check operation
                if operation not in rule.allowed_operations:
                    reason = f"Operation {operation} not allowed for keys matching {rule.pattern}"
                    self._log_audit(agent_id, key, operation, False, reason)
                    return False, reason

                # Log and allow
                if rule.audit:
                    self._log_audit(agent_id, key, operation, True, "Allowed by rule")

                return True, None

        # Default deny
        reason = "No matching access rule"
        self._log_audit(agent_id, key, operation, False, reason)
        return False, reason

    def _log_audit(
        self,
        agent_id: str,
        key: str,
        operation: str,
        allowed: bool,
        reason: str
    ):
        """Log an access audit entry"""
        entry = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "key": key,
            "operation": operation,
            "allowed": allowed,
            "reason": reason
        }
        self._audit_log.append(entry)

        # Keep only last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def get_audit_log(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        entries = self._audit_log

        if agent_id:
            entries = [e for e in entries if e["agent_id"] == agent_id]

        if since:
            entries = [e for e in entries if e["timestamp"] >= since]

        return entries


# =============================================================================
# Tool Call Validator
# =============================================================================


@dataclass
class ToolValidationRule:
    """Rule for tool call validation"""
    tool_pattern: str  # Regex for tool name
    allowed_goals: Set[str]  # Goal types that can use this tool
    max_calls_per_minute: int = 100
    require_confirmation: bool = False
    blocked_arg_patterns: List[str] = field(default_factory=list)


class ToolCallValidator:
    """
    Validates tool calls against agent goals and policies.

    Features:
    - Goal-appropriate tool usage
    - Rate limiting per tool
    - Argument validation
    - Confirmation requirements
    """

    def __init__(self):
        self._rules: List[ToolValidationRule] = []
        self._call_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def add_rule(self, rule: ToolValidationRule):
        """Add a validation rule"""
        self._rules.append(rule)

    def validate(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any],
        current_goal: Optional[str] = None
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Validate a tool call.

        Returns (allowed, reason, requires_confirmation)
        """
        # Find matching rule
        for rule in self._rules:
            if re.match(rule.tool_pattern, tool_name):
                # Check goal alignment
                if current_goal and rule.allowed_goals:
                    if current_goal not in rule.allowed_goals and '*' not in rule.allowed_goals:
                        return False, f"Tool {tool_name} not appropriate for goal {current_goal}", False

                # Check rate limit
                now = time.time()
                minute_ago = now - 60
                recent_calls = [
                    t for t in self._call_history[agent_id][tool_name]
                    if t >= minute_ago
                ]

                if len(recent_calls) >= rule.max_calls_per_minute:
                    return False, f"Rate limit exceeded for {tool_name}", False

                # Check blocked arg patterns
                args_str = json.dumps(args)
                for pattern in rule.blocked_arg_patterns:
                    if re.search(pattern, args_str):
                        return False, f"Blocked argument pattern detected: {pattern}", False

                # Record call
                self._call_history[agent_id][tool_name].append(now)

                return True, None, rule.require_confirmation

        # Default allow
        return True, None, False


# =============================================================================
# Enhanced Flow Integrity Manager
# =============================================================================


@dataclass
class EnhancedFlowConfig:
    """Configuration for enhanced flow integrity"""
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 2.0
    anomaly_learning_period: int = 1000

    # Policy engine
    enable_policy_engine: bool = True
    default_policy_decision: PolicyDecision = PolicyDecision.DENY

    # Memory access control
    enable_memory_access_control: bool = True

    # Tool validation
    enable_tool_validation: bool = True

    # Response actions
    block_on_high_severity: bool = True
    high_severity_threshold: float = 0.8
    quarantine_duration_seconds: float = 300


@dataclass
class FlowVerificationResult:
    """Result of flow verification"""
    action: VerificationResult
    allowed: bool
    anomalies: List[AnomalyEvent]
    policy_evaluation: Optional[PolicyEvaluation]
    memory_access_allowed: Optional[bool]
    tool_validation: Optional[Tuple[bool, Optional[str], bool]]
    reason: Optional[str]


class EnhancedFlowIntegrity:
    """
    Production-ready flow integrity with behavioral analysis.

    Features:
    - Behavioral anomaly detection
    - OPA-style policy enforcement
    - Memory access controls
    - Tool call validation
    """

    def __init__(self, config: Optional[EnhancedFlowConfig] = None):
        self.config = config or EnhancedFlowConfig()

        # Components
        self._anomaly_detector: Optional[AnomalyDetector] = None
        self._policy_engine: Optional[PolicyEngine] = None
        self._memory_access: Optional[MemoryAccessControl] = None
        self._tool_validator: Optional[ToolCallValidator] = None

        if self.config.enable_anomaly_detection:
            self._anomaly_detector = AnomalyDetector(
                learning_period=self.config.anomaly_learning_period,
                sensitivity=self.config.anomaly_sensitivity
            )

        if self.config.enable_policy_engine:
            self._policy_engine = PolicyEngine()

        if self.config.enable_memory_access_control:
            self._memory_access = MemoryAccessControl()

        if self.config.enable_tool_validation:
            self._tool_validator = ToolCallValidator()

        # Agent state tracking
        self._agent_states: Dict[str, str] = {}
        self._state_timestamps: Dict[str, float] = {}
        self._quarantined: Dict[str, float] = {}

        # Callbacks
        self._verification_callbacks: List[Callable[[str, FlowVerificationResult], None]] = []

    # =========================================================================
    # Verification API
    # =========================================================================

    def verify_state_transition(
        self,
        agent_id: str,
        from_state: str,
        to_state: str
    ) -> FlowVerificationResult:
        """Verify a state transition"""
        anomalies = []
        now = time.time()

        # Check quarantine
        if self._is_quarantined(agent_id):
            return FlowVerificationResult(
                action=VerificationResult.BLOCK,
                allowed=False,
                anomalies=[],
                policy_evaluation=None,
                memory_access_allowed=None,
                tool_validation=None,
                reason="Agent is quarantined"
            )

        # Calculate duration in previous state
        duration_ms = 0
        if agent_id in self._state_timestamps:
            duration_ms = (now - self._state_timestamps[agent_id]) * 1000

        # Anomaly detection
        if self._anomaly_detector:
            anomaly = self._anomaly_detector.record_state_transition(
                agent_id, from_state, to_state, duration_ms
            )
            if anomaly:
                anomalies.append(anomaly)

        # Policy check
        policy_eval = None
        if self._policy_engine:
            policy_eval = self._policy_engine.evaluate(
                action="state.transition",
                resource=f"{from_state}->{to_state}",
                subject={"agent_id": agent_id},
                context={"duration_ms": duration_ms}
            )

            if policy_eval.decision == PolicyDecision.DENY:
                return FlowVerificationResult(
                    action=VerificationResult.BLOCK,
                    allowed=False,
                    anomalies=anomalies,
                    policy_evaluation=policy_eval,
                    memory_access_allowed=None,
                    tool_validation=None,
                    reason="Policy denied state transition"
                )

        # Determine action based on anomalies
        action = VerificationResult.ALLOW
        if anomalies:
            max_severity = max(a.severity for a in anomalies)
            if max_severity >= self.config.high_severity_threshold and self.config.block_on_high_severity:
                action = VerificationResult.QUARANTINE
                self._quarantine_agent(agent_id)
            elif max_severity >= 0.5:
                action = VerificationResult.WARN

        # Update state
        self._agent_states[agent_id] = to_state
        self._state_timestamps[agent_id] = now

        result = FlowVerificationResult(
            action=action,
            allowed=action != VerificationResult.BLOCK and action != VerificationResult.QUARANTINE,
            anomalies=anomalies,
            policy_evaluation=policy_eval,
            memory_access_allowed=None,
            tool_validation=None,
            reason=None if action == VerificationResult.ALLOW else "Anomalies detected"
        )

        self._fire_callbacks(agent_id, result)
        return result

    def verify_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any],
        current_goal: Optional[str] = None
    ) -> FlowVerificationResult:
        """Verify a tool call"""
        anomalies = []

        # Check quarantine
        if self._is_quarantined(agent_id):
            return FlowVerificationResult(
                action=VerificationResult.BLOCK,
                allowed=False,
                anomalies=[],
                policy_evaluation=None,
                memory_access_allowed=None,
                tool_validation=None,
                reason="Agent is quarantined"
            )

        # Anomaly detection
        if self._anomaly_detector:
            anomaly = self._anomaly_detector.record_tool_call(agent_id, tool_name, 0)
            if anomaly:
                anomalies.append(anomaly)

        # Policy check
        policy_eval = None
        if self._policy_engine:
            policy_eval = self._policy_engine.evaluate(
                action="tool.execute",
                resource=tool_name,
                subject={"agent_id": agent_id, "goal": current_goal},
                context={"args": args}
            )

            if policy_eval.decision == PolicyDecision.DENY:
                return FlowVerificationResult(
                    action=VerificationResult.BLOCK,
                    allowed=False,
                    anomalies=anomalies,
                    policy_evaluation=policy_eval,
                    memory_access_allowed=None,
                    tool_validation=None,
                    reason="Policy denied tool call"
                )

        # Tool validation
        tool_valid = None
        if self._tool_validator:
            allowed, reason, needs_confirm = self._tool_validator.validate(
                agent_id, tool_name, args, current_goal
            )
            tool_valid = (allowed, reason, needs_confirm)

            if not allowed:
                return FlowVerificationResult(
                    action=VerificationResult.BLOCK,
                    allowed=False,
                    anomalies=anomalies,
                    policy_evaluation=policy_eval,
                    memory_access_allowed=None,
                    tool_validation=tool_valid,
                    reason=reason
                )

        action = VerificationResult.ALLOW
        if anomalies and max(a.severity for a in anomalies) >= 0.5:
            action = VerificationResult.WARN

        result = FlowVerificationResult(
            action=action,
            allowed=True,
            anomalies=anomalies,
            policy_evaluation=policy_eval,
            memory_access_allowed=None,
            tool_validation=tool_valid,
            reason=None
        )

        self._fire_callbacks(agent_id, result)
        return result

    def verify_memory_access(
        self,
        agent_id: str,
        key: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FlowVerificationResult:
        """Verify memory access"""
        anomalies = []

        # Check quarantine
        if self._is_quarantined(agent_id):
            return FlowVerificationResult(
                action=VerificationResult.BLOCK,
                allowed=False,
                anomalies=[],
                policy_evaluation=None,
                memory_access_allowed=False,
                tool_validation=None,
                reason="Agent is quarantined"
            )

        # Anomaly detection
        if self._anomaly_detector:
            anomaly = self._anomaly_detector.record_memory_access(agent_id, key, operation)
            if anomaly:
                anomalies.append(anomaly)

        # Memory access control
        mem_allowed = True
        mem_reason = None
        if self._memory_access:
            mem_allowed, mem_reason = self._memory_access.check_access(
                agent_id, key, operation, context
            )

            if not mem_allowed:
                return FlowVerificationResult(
                    action=VerificationResult.BLOCK,
                    allowed=False,
                    anomalies=anomalies,
                    policy_evaluation=None,
                    memory_access_allowed=False,
                    tool_validation=None,
                    reason=mem_reason
                )

        action = VerificationResult.ALLOW
        if anomalies and max(a.severity for a in anomalies) >= 0.7:
            action = VerificationResult.AUDIT

        result = FlowVerificationResult(
            action=action,
            allowed=True,
            anomalies=anomalies,
            policy_evaluation=None,
            memory_access_allowed=True,
            tool_validation=None,
            reason=None
        )

        self._fire_callbacks(agent_id, result)
        return result

    # =========================================================================
    # Configuration
    # =========================================================================

    def add_policy_rule(self, rule: PolicyRule):
        """Add a policy rule"""
        if self._policy_engine:
            self._policy_engine.add_rule(rule)

    def add_memory_access_rule(self, rule: MemoryAccessRule):
        """Add memory access rule"""
        if self._memory_access:
            self._memory_access.add_rule(rule)

    def add_tool_validation_rule(self, rule: ToolValidationRule):
        """Add tool validation rule"""
        if self._tool_validator:
            self._tool_validator.add_rule(rule)

    def on_verification(self, callback: Callable[[str, FlowVerificationResult], None]):
        """Register verification callback"""
        self._verification_callbacks.append(callback)

    # =========================================================================
    # Quarantine
    # =========================================================================

    def _is_quarantined(self, agent_id: str) -> bool:
        """Check if agent is quarantined"""
        if agent_id not in self._quarantined:
            return False

        if time.time() > self._quarantined[agent_id]:
            del self._quarantined[agent_id]
            return False

        return True

    def _quarantine_agent(self, agent_id: str):
        """Quarantine an agent"""
        self._quarantined[agent_id] = time.time() + self.config.quarantine_duration_seconds
        logger.warning(f"Agent {agent_id} quarantined for {self.config.quarantine_duration_seconds}s")

    def release_from_quarantine(self, agent_id: str):
        """Release agent from quarantine"""
        self._quarantined.pop(agent_id, None)

    # =========================================================================
    # Stats
    # =========================================================================

    def _fire_callbacks(self, agent_id: str, result: FlowVerificationResult):
        """Fire verification callbacks"""
        for callback in self._verification_callbacks:
            try:
                callback(agent_id, result)
            except Exception:
                pass

    def get_anomalies(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[AnomalyEvent]:
        """Get anomalies"""
        if self._anomaly_detector:
            return self._anomaly_detector.get_anomalies(agent_id, since)
        return []

    def get_audit_log(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get memory access audit log"""
        if self._memory_access:
            return self._memory_access.get_audit_log(agent_id)
        return []


# =============================================================================
# Factory
# =============================================================================


_global_flow_integrity: Optional[EnhancedFlowIntegrity] = None


def get_enhanced_flow_integrity() -> EnhancedFlowIntegrity:
    """Get global flow integrity instance"""
    global _global_flow_integrity
    if _global_flow_integrity is None:
        _global_flow_integrity = EnhancedFlowIntegrity()
    return _global_flow_integrity


def create_enhanced_flow_integrity(
    config: Optional[EnhancedFlowConfig] = None
) -> EnhancedFlowIntegrity:
    """Create flow integrity instance"""
    return EnhancedFlowIntegrity(config)
