# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Policy Engine - OPA/SELinux-Inspired Policy System

Inspired by Open Policy Agent (OPA) and SELinux, provides:
- Declarative policy definition (Rego-like DSL)
- Attribute-based access control (ABAC)
- Role-based access control (RBAC)
- Policy evaluation and decision making
- Audit logging
- Policy versioning and hot-reload

Key concepts:
- Policy: A set of rules defining allowed/denied actions
- Rule: A single condition that evaluates to allow/deny
- Subject: Who is performing the action (agent, user)
- Resource: What is being accessed
- Action: What operation is being performed
- Context: Additional information for decision making
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.policy")


# =============================================================================
# Core Types
# =============================================================================

class Decision(Enum):
    """Policy decision result."""
    ALLOW = "allow"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"


class Effect(Enum):
    """Rule effect."""
    ALLOW = "allow"
    DENY = "deny"


class CombiningAlgorithm(Enum):
    """How to combine multiple rule results."""
    DENY_OVERRIDES = "deny_overrides"        # Any deny -> deny
    PERMIT_OVERRIDES = "permit_overrides"    # Any permit -> permit
    FIRST_APPLICABLE = "first_applicable"    # First matching rule wins
    DENY_UNLESS_PERMIT = "deny_unless_permit"  # Default deny


# =============================================================================
# Policy Request/Response
# =============================================================================

@dataclass
class Subject:
    """Who is performing the action."""

    id: str
    type: str = "agent"  # agent, user, service, system

    # Attributes
    roles: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    # Trust level
    trust_level: int = 0  # 0=untrusted, 1=basic, 2=verified, 3=trusted

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_label(self, key: str, value: Optional[str] = None) -> bool:
        if value is None:
            return key in self.labels
        return self.labels.get(key) == value


@dataclass
class Resource:
    """What is being accessed."""

    id: str
    type: str  # adapter, workflow, state, file, network, etc.

    # Attributes
    owner: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    path: Optional[str] = None
    namespace: str = "default"

    # Sensitivity
    classification: str = "public"  # public, internal, confidential, secret


@dataclass
class Action:
    """What operation is being performed."""

    name: str
    category: str = "general"  # read, write, execute, admin

    # Method details
    method: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Additional context for policy evaluation."""

    # Time
    timestamp: datetime = field(default_factory=datetime.now)

    # Environment
    environment: str = "production"  # development, staging, production

    # Request metadata
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRequest:
    """A policy evaluation request."""

    subject: Subject
    resource: Resource
    action: Action
    context: Context = field(default_factory=Context)


@dataclass
class PolicyResponse:
    """Result of policy evaluation."""

    decision: Decision
    reason: str = ""

    # Which rules matched
    matched_rules: List[str] = field(default_factory=list)

    # Obligations (must be fulfilled)
    obligations: List[Dict[str, Any]] = field(default_factory=list)

    # Advice (optional recommendations)
    advice: List[str] = field(default_factory=list)

    # Audit info
    evaluation_time_ms: float = 0
    policy_version: str = ""


# =============================================================================
# Rules and Conditions
# =============================================================================

class Condition(ABC):
    """Base class for policy conditions."""

    @abstractmethod
    def evaluate(self, request: PolicyRequest) -> bool:
        """Evaluate the condition against a request."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass


class AttributeCondition(Condition):
    """Condition based on attribute comparison."""

    def __init__(
        self,
        path: str,           # e.g., "subject.roles", "resource.type"
        operator: str,       # eq, ne, in, contains, matches, gt, lt
        value: Any
    ):
        self.path = path
        self.operator = operator
        self.value = value

    def evaluate(self, request: PolicyRequest) -> bool:
        # Get attribute value from request
        actual = self._get_value(request)

        # Compare
        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "contains":
            return self.value in actual if isinstance(actual, (list, str, dict)) else False
        elif self.operator == "matches":
            return bool(re.match(self.value, str(actual)))
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value

        return False

    def _get_value(self, request: PolicyRequest) -> Any:
        """Get value from request using path."""
        parts = self.path.split(".")
        obj: Any = request

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None

        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "attribute",
            "path": self.path,
            "operator": self.operator,
            "value": self.value,
        }


class CompositeCondition(Condition):
    """Composite condition (AND, OR, NOT)."""

    def __init__(
        self,
        operator: str,  # and, or, not
        conditions: List[Condition]
    ):
        self.operator = operator
        self.conditions = conditions

    def evaluate(self, request: PolicyRequest) -> bool:
        if self.operator == "and":
            return all(c.evaluate(request) for c in self.conditions)
        elif self.operator == "or":
            return any(c.evaluate(request) for c in self.conditions)
        elif self.operator == "not":
            return not self.conditions[0].evaluate(request) if self.conditions else True

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "composite",
            "operator": self.operator,
            "conditions": [c.to_dict() for c in self.conditions],
        }


class TimeCondition(Condition):
    """Time-based condition."""

    def __init__(
        self,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        days_of_week: Optional[List[int]] = None,  # 0=Monday, 6=Sunday
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.days_of_week = days_of_week

    def evaluate(self, request: PolicyRequest) -> bool:
        now = request.context.timestamp

        if self.days_of_week is not None:
            if now.weekday() not in self.days_of_week:
                return False

        if self.start_hour is not None and now.hour < self.start_hour:
            return False

        if self.end_hour is not None and now.hour >= self.end_hour:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "time",
            "startHour": self.start_hour,
            "endHour": self.end_hour,
            "daysOfWeek": self.days_of_week,
        }


class ExpressionCondition(Condition):
    """Expression-based condition (simple Rego-like expressions)."""

    def __init__(self, expression: str):
        self.expression = expression
        self._compiled = self._compile(expression)

    def _compile(self, expr: str) -> Callable[[PolicyRequest], bool]:
        """Compile expression to callable."""
        # Simple expression parser
        # Supports: subject.X, resource.X, action.X, context.X
        # Operators: ==, !=, in, contains

        def evaluator(request: PolicyRequest) -> bool:
            try:
                # Very simple eval - in production would use proper parser
                # This is just for demonstration
                local_vars = {
                    "subject": request.subject,
                    "resource": request.resource,
                    "action": request.action,
                    "context": request.context,
                }

                # Safety: only allow attribute access
                if "(" in expr or "import" in expr or "__" in expr:
                    return False

                return eval(expr, {"__builtins__": {}}, local_vars)
            except Exception:
                return False

        return evaluator

    def evaluate(self, request: PolicyRequest) -> bool:
        return self._compiled(request)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "expression",
            "expression": self.expression,
        }


# =============================================================================
# Rule Definition
# =============================================================================

@dataclass
class Rule:
    """A policy rule."""

    id: str
    effect: Effect
    description: str = ""

    # Targeting
    subjects: List[str] = field(default_factory=list)      # Subject patterns
    resources: List[str] = field(default_factory=list)     # Resource patterns
    actions: List[str] = field(default_factory=list)       # Action patterns

    # Conditions
    conditions: List[Condition] = field(default_factory=list)

    # Obligations
    obligations: List[Dict[str, Any]] = field(default_factory=list)

    # Priority (higher = evaluated first)
    priority: int = 0

    # Enabled
    enabled: bool = True

    def matches_request(self, request: PolicyRequest) -> bool:
        """Check if this rule applies to the request."""
        # Check subject
        if self.subjects:
            if not self._matches_pattern(request.subject.id, self.subjects):
                if not self._matches_pattern(request.subject.type, self.subjects):
                    return False

        # Check resource
        if self.resources:
            if not self._matches_pattern(request.resource.id, self.resources):
                if not self._matches_pattern(request.resource.type, self.resources):
                    return False

        # Check action
        if self.actions:
            if not self._matches_pattern(request.action.name, self.actions):
                return False

        return True

    def _matches_pattern(self, value: str, patterns: List[str]) -> bool:
        """Check if value matches any pattern."""
        for pattern in patterns:
            if pattern == "*":
                return True
            if pattern == value:
                return True
            if pattern.endswith("*") and value.startswith(pattern[:-1]):
                return True
            if pattern.startswith("*") and value.endswith(pattern[1:]):
                return True

        return False

    def evaluate_conditions(self, request: PolicyRequest) -> bool:
        """Evaluate all conditions."""
        return all(c.evaluate(request) for c in self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "effect": self.effect.value,
            "description": self.description,
            "subjects": self.subjects,
            "resources": self.resources,
            "actions": self.actions,
            "conditions": [c.to_dict() for c in self.conditions],
            "priority": self.priority,
            "enabled": self.enabled,
        }


# =============================================================================
# Policy Definition
# =============================================================================

@dataclass
class Policy:
    """A collection of rules."""

    id: str
    name: str
    version: str = "1.0"
    description: str = ""

    # Rules
    rules: List[Rule] = field(default_factory=list)

    # Combining algorithm
    combining_algorithm: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES

    # Default decision when no rules match
    default_decision: Decision = Decision.DENY

    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Enabled
    enabled: bool = True

    def add_rule(self, rule: Rule):
        """Add a rule to this policy."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.updated_at = datetime.now()

    def evaluate(self, request: PolicyRequest) -> PolicyResponse:
        """Evaluate the policy against a request."""
        start_time = datetime.now()

        matched_rules = []
        decisions = []
        obligations = []
        advice = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            if not rule.matches_request(request):
                continue

            if not rule.evaluate_conditions(request):
                continue

            matched_rules.append(rule.id)
            decisions.append(rule.effect)
            obligations.extend(rule.obligations)

            # Apply combining algorithm
            if self.combining_algorithm == CombiningAlgorithm.FIRST_APPLICABLE:
                break
            elif self.combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
                if rule.effect == Effect.DENY:
                    break
            elif self.combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
                if rule.effect == Effect.ALLOW:
                    break

        # Determine final decision
        if not decisions:
            decision = self.default_decision
            reason = "No matching rules"
        else:
            if self.combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
                decision = Decision.DENY if Effect.DENY in decisions else Decision.ALLOW
            elif self.combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
                decision = Decision.ALLOW if Effect.ALLOW in decisions else Decision.DENY
            elif self.combining_algorithm == CombiningAlgorithm.DENY_UNLESS_PERMIT:
                decision = Decision.ALLOW if Effect.ALLOW in decisions else Decision.DENY
            else:
                decision = Decision.ALLOW if decisions[0] == Effect.ALLOW else Decision.DENY

            reason = f"Matched {len(matched_rules)} rules"

        eval_time = (datetime.now() - start_time).total_seconds() * 1000

        return PolicyResponse(
            decision=decision,
            reason=reason,
            matched_rules=matched_rules,
            obligations=obligations,
            advice=advice,
            evaluation_time_ms=eval_time,
            policy_version=self.version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "combiningAlgorithm": self.combining_algorithm.value,
            "defaultDecision": self.default_decision.value,
            "author": self.author,
            "enabled": self.enabled,
        }


# =============================================================================
# Policy Set
# =============================================================================

@dataclass
class PolicySet:
    """A set of policies."""

    id: str
    name: str
    policies: List[Policy] = field(default_factory=list)

    # Combining algorithm for policies
    combining_algorithm: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES

    def add_policy(self, policy: Policy):
        """Add a policy to this set."""
        self.policies.append(policy)

    def evaluate(self, request: PolicyRequest) -> PolicyResponse:
        """Evaluate all policies."""
        all_responses = []

        for policy in self.policies:
            if not policy.enabled:
                continue
            response = policy.evaluate(request)
            if response.decision != Decision.NOT_APPLICABLE:
                all_responses.append(response)

        if not all_responses:
            return PolicyResponse(
                decision=Decision.DENY,
                reason="No applicable policies",
            )

        # Combine responses
        decisions = [r.decision for r in all_responses]
        matched_rules = []
        obligations = []

        for r in all_responses:
            matched_rules.extend(r.matched_rules)
            obligations.extend(r.obligations)

        if self.combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
            final = Decision.DENY if Decision.DENY in decisions else Decision.ALLOW
        elif self.combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
            final = Decision.ALLOW if Decision.ALLOW in decisions else Decision.DENY
        else:
            final = decisions[0]

        return PolicyResponse(
            decision=final,
            reason=f"Combined {len(all_responses)} policy results",
            matched_rules=matched_rules,
            obligations=obligations,
        )


# =============================================================================
# RBAC Support
# =============================================================================

@dataclass
class Role:
    """A role in RBAC."""

    name: str
    description: str = ""

    # Permissions
    permissions: List[str] = field(default_factory=list)

    # Inheritance
    inherits: List[str] = field(default_factory=list)


@dataclass
class Permission:
    """A permission in RBAC."""

    name: str
    resource_type: str
    actions: List[str] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)


class RBACEngine:
    """Role-Based Access Control engine."""

    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._permissions: Dict[str, Permission] = {}
        self._role_assignments: Dict[str, Set[str]] = defaultdict(set)

    def add_role(self, role: Role):
        """Add a role."""
        self._roles[role.name] = role

    def add_permission(self, permission: Permission):
        """Add a permission."""
        self._permissions[permission.name] = permission

    def assign_role(self, subject_id: str, role_name: str):
        """Assign a role to a subject."""
        self._role_assignments[subject_id].add(role_name)

    def revoke_role(self, subject_id: str, role_name: str):
        """Revoke a role from a subject."""
        self._role_assignments[subject_id].discard(role_name)

    def get_roles(self, subject_id: str) -> Set[str]:
        """Get all roles for a subject (including inherited)."""
        roles = self._role_assignments[subject_id].copy()

        # Add inherited roles
        to_check = list(roles)
        while to_check:
            role_name = to_check.pop()
            role = self._roles.get(role_name)
            if role:
                for inherited in role.inherits:
                    if inherited not in roles:
                        roles.add(inherited)
                        to_check.append(inherited)

        return roles

    def get_permissions(self, subject_id: str) -> Set[str]:
        """Get all permissions for a subject."""
        permissions = set()

        for role_name in self.get_roles(subject_id):
            role = self._roles.get(role_name)
            if role:
                permissions.update(role.permissions)

        return permissions

    def check_permission(
        self,
        subject_id: str,
        resource_type: str,
        action: str
    ) -> bool:
        """Check if subject has permission for action on resource type."""
        for perm_name in self.get_permissions(subject_id):
            perm = self._permissions.get(perm_name)
            if perm and perm.resource_type == resource_type:
                if action in perm.actions or "*" in perm.actions:
                    return True

        return False


# =============================================================================
# Audit Log
# =============================================================================

@dataclass
class AuditEntry:
    """An entry in the audit log."""

    timestamp: datetime
    request_id: str
    subject_id: str
    resource_id: str
    action: str
    decision: Decision
    matched_rules: List[str]
    evaluation_time_ms: float

    # Additional context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    reason: str = ""


class AuditLogger:
    """Audit logging for policy decisions."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self._entries: List[AuditEntry] = []
        self._max_entries = 10000

    def log(
        self,
        request: PolicyRequest,
        response: PolicyResponse
    ):
        """Log a policy decision."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            request_id=request.context.request_id or str(id(request)),
            subject_id=request.subject.id,
            resource_id=request.resource.id,
            action=request.action.name,
            decision=response.decision,
            matched_rules=response.matched_rules,
            evaluation_time_ms=response.evaluation_time_ms,
            source_ip=request.context.source_ip,
            user_agent=request.context.user_agent,
            reason=response.reason,
        )

        self._entries.append(entry)

        # Trim if too many
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Persist if configured
        if self.log_path:
            self._persist_entry(entry)

        # Log important decisions
        if response.decision == Decision.DENY:
            logger.warning(
                f"DENY: {request.subject.id} -> {request.resource.id}:{request.action.name} "
                f"({response.reason})"
            )

    def _persist_entry(self, entry: AuditEntry):
        """Persist entry to disk."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp.isoformat(),
                "requestId": entry.request_id,
                "subjectId": entry.subject_id,
                "resourceId": entry.resource_id,
                "action": entry.action,
                "decision": entry.decision.value,
                "matchedRules": entry.matched_rules,
                "reason": entry.reason,
            }) + "\n")

    def query(
        self,
        subject_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        decision: Optional[Decision] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Query audit entries."""
        results = []

        for entry in reversed(self._entries):
            if subject_id and entry.subject_id != subject_id:
                continue
            if resource_id and entry.resource_id != resource_id:
                continue
            if decision and entry.decision != decision:
                continue
            if since and entry.timestamp < since:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results


# =============================================================================
# Policy Engine
# =============================================================================

class PolicyEngine:
    """
    Central policy engine.

    Handles:
    - Policy storage and management
    - Policy evaluation
    - RBAC integration
    - Audit logging
    - Hot reload
    """

    def __init__(
        self,
        policies_path: Optional[Path] = None,
        audit_path: Optional[Path] = None
    ):
        self.policies_path = policies_path or (Path.home() / ".bbx" / "policies")
        self.policies_path.mkdir(parents=True, exist_ok=True)

        # Policy storage
        self._policies: Dict[str, Policy] = {}
        self._policy_sets: Dict[str, PolicySet] = {}

        # RBAC
        self.rbac = RBACEngine()

        # Audit
        self.audit = AuditLogger(audit_path)

        # Caching
        self._decision_cache: Dict[str, Tuple[PolicyResponse, datetime]] = {}
        self._cache_ttl_seconds = 60

        # Statistics
        self._stats = {
            "evaluations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "allows": 0,
            "denies": 0,
        }

        # Load built-in policies
        self._load_builtin_policies()

    def _load_builtin_policies(self):
        """Load built-in policies."""
        # Default policy
        default = Policy(
            id="default",
            name="Default Policy",
            description="Default allow/deny rules",
            default_decision=Decision.DENY,
        )

        # Allow system operations
        default.add_rule(Rule(
            id="allow-system",
            effect=Effect.ALLOW,
            subjects=["system:*"],
            resources=["*"],
            actions=["*"],
            priority=1000,
        ))

        # Allow read operations for verified subjects
        default.add_rule(Rule(
            id="allow-verified-read",
            effect=Effect.ALLOW,
            subjects=["*"],
            resources=["*"],
            actions=["read", "list", "get"],
            conditions=[
                AttributeCondition("subject.trust_level", "gte", 1)
            ],
            priority=100,
        ))

        self.add_policy(default)

        # Adapter access policy
        adapter_policy = Policy(
            id="adapter-access",
            name="Adapter Access Policy",
            description="Controls access to adapters",
        )

        # Allow LLM adapter for all verified
        adapter_policy.add_rule(Rule(
            id="allow-llm-adapter",
            effect=Effect.ALLOW,
            resources=["adapter:llm", "adapter:openai", "adapter:anthropic"],
            actions=["execute"],
            conditions=[
                AttributeCondition("subject.trust_level", "gte", 1)
            ],
        ))

        # Restrict shell adapter
        adapter_policy.add_rule(Rule(
            id="restrict-shell-adapter",
            effect=Effect.DENY,
            resources=["adapter:shell", "adapter:bash"],
            actions=["execute"],
            conditions=[
                AttributeCondition("subject.trust_level", "lt", 2)
            ],
        ))

        self.add_policy(adapter_policy)

    def add_policy(self, policy: Policy):
        """Add a policy."""
        self._policies[policy.id] = policy
        self._invalidate_cache()

    def remove_policy(self, policy_id: str):
        """Remove a policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            self._invalidate_cache()

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(self) -> List[Policy]:
        """List all policies."""
        return list(self._policies.values())

    def add_policy_set(self, policy_set: PolicySet):
        """Add a policy set."""
        self._policy_sets[policy_set.id] = policy_set

    async def evaluate(
        self,
        request: PolicyRequest,
        use_cache: bool = True
    ) -> PolicyResponse:
        """Evaluate a policy request."""
        self._stats["evaluations"] += 1

        # Check cache
        cache_key = self._get_cache_key(request)
        if use_cache and cache_key in self._decision_cache:
            cached, cached_at = self._decision_cache[cache_key]
            if (datetime.now() - cached_at).total_seconds() < self._cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return cached

        self._stats["cache_misses"] += 1

        # Evaluate all applicable policies
        responses = []

        for policy in self._policies.values():
            if policy.enabled:
                response = policy.evaluate(request)
                if response.decision != Decision.NOT_APPLICABLE:
                    responses.append(response)

        # Combine responses
        if not responses:
            final_response = PolicyResponse(
                decision=Decision.DENY,
                reason="No applicable policies",
            )
        else:
            # Default to deny-overrides
            decisions = [r.decision for r in responses]
            if Decision.DENY in decisions:
                final_response = PolicyResponse(
                    decision=Decision.DENY,
                    reason="Denied by policy",
                    matched_rules=[r for resp in responses for r in resp.matched_rules],
                )
            else:
                final_response = PolicyResponse(
                    decision=Decision.ALLOW,
                    reason="Allowed by policy",
                    matched_rules=[r for resp in responses for r in resp.matched_rules],
                )

        # Update stats
        if final_response.decision == Decision.ALLOW:
            self._stats["allows"] += 1
        else:
            self._stats["denies"] += 1

        # Cache result
        if use_cache:
            self._decision_cache[cache_key] = (final_response, datetime.now())

        # Audit log
        self.audit.log(request, final_response)

        return final_response

    def _get_cache_key(self, request: PolicyRequest) -> str:
        """Generate cache key for a request."""
        key_data = f"{request.subject.id}:{request.resource.id}:{request.action.name}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _invalidate_cache(self):
        """Invalidate the decision cache."""
        self._decision_cache.clear()

    async def check(
        self,
        subject_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        **context_attrs
    ) -> bool:
        """Convenience method to check a permission."""
        request = PolicyRequest(
            subject=Subject(id=subject_id),
            resource=Resource(id=resource_id, type=resource_type),
            action=Action(name=action),
            context=Context(attributes=context_attrs),
        )

        response = await self.evaluate(request)
        return response.decision == Decision.ALLOW

    def save_policy(self, policy: Policy, path: Optional[Path] = None):
        """Save a policy to disk."""
        path = path or (self.policies_path / f"{policy.id}.json")
        with open(path, "w") as f:
            json.dump(policy.to_dict(), f, indent=2)

    def load_policy(self, path: Path) -> Policy:
        """Load a policy from disk."""
        with open(path) as f:
            data = json.load(f)

        policy = Policy(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            combining_algorithm=CombiningAlgorithm(data.get("combiningAlgorithm", "deny_overrides")),
            default_decision=Decision(data.get("defaultDecision", "deny")),
            author=data.get("author", ""),
            enabled=data.get("enabled", True),
        )

        for rule_data in data.get("rules", []):
            rule = Rule(
                id=rule_data["id"],
                effect=Effect(rule_data["effect"]),
                description=rule_data.get("description", ""),
                subjects=rule_data.get("subjects", []),
                resources=rule_data.get("resources", []),
                actions=rule_data.get("actions", []),
                priority=rule_data.get("priority", 0),
                enabled=rule_data.get("enabled", True),
            )
            policy.add_rule(rule)

        return policy

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "policies_count": len(self._policies),
            "cache_size": len(self._decision_cache),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_global_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = PolicyEngine()
    return _global_engine


async def check_permission(
    subject_id: str,
    resource_type: str,
    resource_id: str,
    action: str,
    **context
) -> bool:
    """Check if an action is allowed."""
    engine = get_policy_engine()
    return await engine.check(
        subject_id, resource_type, resource_id, action, **context
    )


def create_policy(
    policy_id: str,
    name: str,
    rules: Optional[List[Dict[str, Any]]] = None
) -> Policy:
    """Create a new policy."""
    policy = Policy(id=policy_id, name=name)

    if rules:
        for rule_data in rules:
            rule = Rule(
                id=rule_data.get("id", f"rule-{len(policy.rules)}"),
                effect=Effect(rule_data.get("effect", "deny")),
                subjects=rule_data.get("subjects", []),
                resources=rule_data.get("resources", []),
                actions=rule_data.get("actions", []),
            )
            policy.add_rule(rule)

    return policy
