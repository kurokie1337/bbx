# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Intent Engine - Semantic Workflow Compression/Decompression

The true nature of .bbx files: compressed intentions that expand into workflows.

Instead of writing 100 lines of YAML, you write:
    intent: "Deploy React app to AWS S3"

And BBX expands it into a full workflow using:
    1. RAG - Find similar successful workflows from memory
    2. LLM - Generate concrete steps
    3. Adapters - Execute

This is SEMANTIC COMPRESSION for DevOps.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                        .bbx Intent File                        │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ bbx: 2.0                                                 │  │
    │  │ intent: "Deploy to production with zero downtime"        │  │
    │  │ constraints: [no-downtime, cost < $50]                   │  │
    │  │ hints: [use docker, prefer aws]                          │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └───────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                      Intent Engine                              │
    │  ┌────────────┐   ┌────────────┐   ┌────────────────────────┐  │
    │  │ Understand │ → │ RAG Search │ → │ Generate Workflow      │  │
    │  │ Intent     │   │ Memory     │   │ (LLM + Templates)      │  │
    │  └────────────┘   └────────────┘   └────────────────────────┘  │
    └───────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                    Expanded Workflow (DAG)                      │
    │  build → test → deploy → verify → notify                       │
    └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("bbx.intent")


class IntentType(Enum):
    """Types of intents"""
    DEPLOY = "deploy"
    BUILD = "build"
    TEST = "test"
    MONITOR = "monitor"
    BACKUP = "backup"
    MIGRATE = "migrate"
    SCALE = "scale"
    CUSTOM = "custom"


@dataclass
class Constraint:
    """A constraint on workflow execution"""
    type: str  # 'cost', 'time', 'resource', 'policy'
    operator: str  # '<', '>', '=', 'must', 'should'
    value: Any
    priority: int = 1  # 1 = must, 2 = should, 3 = nice-to-have


@dataclass
class BBXIntent:
    """
    A semantic intent that expands into a workflow.

    This is the TRUE .bbx format - compressed DevOps intention.
    """
    version: str = "2.0"
    intent: str = ""  # Natural language intent
    intent_type: IntentType = IntentType.CUSTOM

    # Context
    target: Optional[str] = None  # What to act on (app, service, infra)
    environment: Optional[str] = None  # dev, staging, prod

    # Constraints
    constraints: List[Constraint] = field(default_factory=list)

    # Hints for execution
    hints: List[str] = field(default_factory=list)

    # Explicit steps (optional - if you want to override)
    explicit_steps: List[Dict] = field(default_factory=list)

    # Metadata
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Embedding (computed)
    embedding: Optional[List[float]] = None

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BBXIntent":
        """Parse intent from YAML"""
        data = yaml.safe_load(yaml_str)

        # Handle both flat and nested formats
        if "intent" in data and isinstance(data["intent"], dict):
            data = data["intent"]

        # Parse constraints
        constraints = []
        for c in data.get("constraints", []):
            if isinstance(c, str):
                # Parse string constraint like "cost < $50"
                constraint = cls._parse_constraint_string(c)
                if constraint:
                    constraints.append(constraint)
            elif isinstance(c, dict):
                constraints.append(Constraint(**c))

        # Detect intent type
        intent_text = data.get("intent", "")
        intent_type = cls._detect_intent_type(intent_text)

        return cls(
            version=str(data.get("bbx", data.get("version", "2.0"))),
            intent=intent_text,
            intent_type=intent_type,
            target=data.get("target"),
            environment=data.get("environment", data.get("env")),
            constraints=constraints,
            hints=data.get("hints", []),
            explicit_steps=data.get("steps", []),
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_file(cls, path: str) -> "BBXIntent":
        """Load intent from file"""
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_yaml(content)

    @staticmethod
    def _parse_constraint_string(s: str) -> Optional[Constraint]:
        """Parse constraint from string like 'cost < $50'"""
        patterns = [
            (r"(\w+)\s*<\s*\$?(\d+)", "cost", "<"),
            (r"(\w+)\s*>\s*\$?(\d+)", "cost", ">"),
            (r"time\s*<\s*(\d+)\s*(min|hour|sec)", "time", "<"),
            (r"no[- ](\w+)", "policy", "must"),
            (r"must[- ](\w+)", "policy", "must"),
            (r"prefer[- ](\w+)", "hint", "should"),
        ]

        for pattern, ctype, op in patterns:
            match = re.search(pattern, s.lower())
            if match:
                return Constraint(
                    type=ctype,
                    operator=op,
                    value=match.groups(),
                    priority=1 if op == "must" else 2,
                )
        return None

    @staticmethod
    def _detect_intent_type(intent: str) -> IntentType:
        """Detect intent type from text"""
        intent_lower = intent.lower()

        type_keywords = {
            IntentType.DEPLOY: ["deploy", "release", "ship", "publish", "push to"],
            IntentType.BUILD: ["build", "compile", "package", "create artifact"],
            IntentType.TEST: ["test", "verify", "validate", "check", "lint"],
            IntentType.MONITOR: ["monitor", "observe", "watch", "alert", "log"],
            IntentType.BACKUP: ["backup", "snapshot", "archive", "save state"],
            IntentType.MIGRATE: ["migrate", "move", "transfer", "upgrade db"],
            IntentType.SCALE: ["scale", "resize", "expand", "add instance"],
        }

        for itype, keywords in type_keywords.items():
            if any(kw in intent_lower for kw in keywords):
                return itype

        return IntentType.CUSTOM

    def to_yaml(self) -> str:
        """Serialize to YAML"""
        data = {
            "bbx": self.version,
            "intent": self.intent,
        }

        if self.target:
            data["target"] = self.target
        if self.environment:
            data["environment"] = self.environment
        if self.constraints:
            data["constraints"] = [
                f"{c.type} {c.operator} {c.value}" for c in self.constraints
            ]
        if self.hints:
            data["hints"] = self.hints
        if self.explicit_steps:
            data["steps"] = self.explicit_steps
        if self.tags:
            data["tags"] = self.tags

        return yaml.dump(data, default_flow_style=False, allow_unicode=True)


@dataclass
class ExpandedWorkflow:
    """A workflow expanded from an intent"""
    intent: BBXIntent
    steps: List[Dict[str, Any]]
    confidence: float  # How confident we are in this expansion
    sources: List[str]  # Where steps came from (memory, template, generated)
    warnings: List[str] = field(default_factory=list)


class IntentEngine:
    """
    Expands intents into executable workflows.

    The "decompressor" for semantic .bbx files.
    """

    def __init__(
        self,
        memory=None,  # SemanticMemory for RAG
        llm=None,     # LLM for generation
    ):
        self.memory = memory
        self.llm = llm
        self._embedder = None
        self._workflow_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load built-in workflow templates by intent type"""
        return {
            "deploy": [
                {"id": "build", "use": "docker.build", "args": {"context": "."}},
                {"id": "push", "use": "docker.push", "args": {"registry": "${REGISTRY}"}, "depends_on": ["build"]},
                {"id": "deploy", "use": "kubernetes.apply", "args": {"manifest": "k8s/"}, "depends_on": ["push"]},
                {"id": "verify", "use": "http.check", "args": {"url": "${APP_URL}"}, "depends_on": ["deploy"]},
            ],
            "build": [
                {"id": "install", "use": "shell.run", "args": {"cmd": "npm install"}},
                {"id": "lint", "use": "shell.run", "args": {"cmd": "npm run lint"}, "depends_on": ["install"]},
                {"id": "build", "use": "shell.run", "args": {"cmd": "npm run build"}, "depends_on": ["lint"]},
            ],
            "test": [
                {"id": "install", "use": "shell.run", "args": {"cmd": "npm install"}},
                {"id": "test", "use": "shell.run", "args": {"cmd": "npm test"}, "depends_on": ["install"]},
            ],
            "backup": [
                {"id": "snapshot", "use": "aws.rds_snapshot", "args": {"db": "${DB_ID}"}},
                {"id": "upload", "use": "aws.s3_upload", "args": {"bucket": "${BACKUP_BUCKET}"}, "depends_on": ["snapshot"]},
            ],
        }

    async def _get_embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not available")
        return self._embedder

    async def _embed_intent(self, intent: BBXIntent) -> List[float]:
        """Create embedding for intent"""
        embedder = await self._get_embedder()
        if embedder is None:
            return []

        # Combine intent text with context
        text_parts = [intent.intent]
        if intent.target:
            text_parts.append(f"target: {intent.target}")
        if intent.environment:
            text_parts.append(f"env: {intent.environment}")
        text_parts.extend(intent.hints)

        text = " | ".join(text_parts)
        embedding = embedder.encode(text, show_progress_bar=False)
        return embedding.tolist()

    async def _search_similar_workflows(
        self,
        intent: BBXIntent,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memory for similar successful workflows"""
        if self.memory is None:
            return []

        try:
            # Search by intent text
            results = await self.memory.recall(
                agent_id="workflows",
                query=intent.intent,
                top_k=top_k,
            )

            similar = []
            for r in results:
                if hasattr(r, 'entry'):
                    content = r.entry.content
                    metadata = r.entry.metadata
                    score = r.score
                else:
                    continue

                # Try to parse as workflow
                try:
                    workflow_data = yaml.safe_load(content)
                    if workflow_data and "steps" in workflow_data:
                        similar.append({
                            "workflow": workflow_data,
                            "score": score,
                            "source": metadata.get("source", "memory"),
                        })
                except:
                    pass

            return similar

        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    async def _generate_with_llm(
        self,
        intent: BBXIntent,
        context: List[Dict] = None,
    ) -> List[Dict]:
        """Generate workflow steps using LLM"""
        if self.llm is None:
            return []

        # Build prompt
        prompt = f"""Generate a BBX workflow for this intent.

Intent: {intent.intent}
Target: {intent.target or 'not specified'}
Environment: {intent.environment or 'not specified'}
Constraints: {[f"{c.type} {c.operator} {c.value}" for c in intent.constraints]}
Hints: {intent.hints}

Output ONLY valid YAML steps in this format:
steps:
  - id: step_name
    use: adapter.method
    args:
      key: value
    depends_on: [previous_step]

Generate practical, production-ready steps:"""

        if context:
            prompt += f"\n\nSimilar workflows for reference:\n"
            for ctx in context[:2]:
                prompt += yaml.dump(ctx["workflow"]["steps"][:3]) + "\n"

        try:
            result = await self.llm(prompt)

            # Parse YAML from response
            yaml_match = re.search(r"steps:\s*\n([\s\S]+?)(?:\n\n|$)", result)
            if yaml_match:
                steps_yaml = "steps:\n" + yaml_match.group(1)
                parsed = yaml.safe_load(steps_yaml)
                return parsed.get("steps", [])

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")

        return []

    def _apply_constraints(
        self,
        steps: List[Dict],
        constraints: List[Constraint],
    ) -> Tuple[List[Dict], List[str]]:
        """Apply constraints to workflow steps"""
        warnings = []

        for constraint in constraints:
            if constraint.type == "policy" and constraint.operator == "must":
                policy = str(constraint.value)

                if "no-downtime" in policy or "zero-downtime" in policy:
                    # Add rolling deployment
                    for step in steps:
                        if "deploy" in step.get("id", "").lower():
                            step.setdefault("args", {})["strategy"] = "rolling"

                if "no-delete" in policy:
                    # Remove any delete steps
                    steps = [s for s in steps if "delete" not in s.get("id", "").lower()]

            elif constraint.type == "cost":
                # Add cost warnings
                warnings.append(f"Cost constraint: {constraint.operator} {constraint.value}")

        return steps, warnings

    def _merge_with_template(
        self,
        intent: BBXIntent,
        similar: List[Dict],
        generated: List[Dict],
    ) -> List[Dict]:
        """Merge template, similar workflows, and generated steps"""
        # Start with template
        template_key = intent.intent_type.value
        base_steps = self._workflow_templates.get(template_key, [])

        if intent.explicit_steps:
            # User provided explicit steps - use those
            return intent.explicit_steps

        if similar and similar[0]["score"] > 0.8:
            # High-confidence match - use from memory
            return similar[0]["workflow"].get("steps", base_steps)

        if generated:
            # Use LLM generated steps
            return generated

        # Fall back to template
        return base_steps

    async def expand(self, intent: BBXIntent) -> ExpandedWorkflow:
        """
        Expand intent into a full workflow.

        This is the main "decompression" function.

        Args:
            intent: The semantic intent

        Returns:
            Expanded workflow ready for execution
        """
        logger.info(f"Expanding intent: {intent.intent[:50]}...")

        sources = []

        # 1. Search for similar workflows in memory (RAG)
        similar = await self._search_similar_workflows(intent)
        if similar:
            sources.append(f"memory ({len(similar)} similar)")
            logger.info(f"Found {len(similar)} similar workflows in memory")

        # 2. Generate with LLM if available
        generated = await self._generate_with_llm(intent, similar)
        if generated:
            sources.append("llm")
            logger.info(f"LLM generated {len(generated)} steps")

        # 3. Merge with templates
        steps = self._merge_with_template(intent, similar, generated)
        if not sources or steps == self._workflow_templates.get(intent.intent_type.value, []):
            sources.append("template")

        # 4. Apply constraints
        steps, warnings = self._apply_constraints(steps, intent.constraints)

        # 5. Calculate confidence
        confidence = 0.5  # Base
        if similar and similar[0]["score"] > 0.8:
            confidence = 0.9
        elif generated:
            confidence = 0.7
        elif steps:
            confidence = 0.6

        return ExpandedWorkflow(
            intent=intent,
            steps=steps,
            confidence=confidence,
            sources=sources,
            warnings=warnings,
        )

    async def expand_file(self, path: str) -> ExpandedWorkflow:
        """Expand intent from file"""
        intent = BBXIntent.from_file(path)
        return await self.expand(intent)

    def to_executable_yaml(self, expanded: ExpandedWorkflow) -> str:
        """Convert expanded workflow to executable BBX YAML"""
        workflow = {
            "workflow": {
                "id": expanded.intent.id or hashlib.md5(
                    expanded.intent.intent.encode()
                ).hexdigest()[:8],
                "name": expanded.intent.name or expanded.intent.intent[:50],
                "description": f"Expanded from intent: {expanded.intent.intent}",
                "steps": expanded.steps,
            }
        }

        # Add metadata
        if expanded.intent.environment:
            workflow["workflow"]["environment"] = expanded.intent.environment

        header = f"""# BBX Workflow (expanded from intent)
# Original intent: {expanded.intent.intent}
# Confidence: {expanded.confidence:.0%}
# Sources: {', '.join(expanded.sources)}
# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

"""
        return header + yaml.dump(workflow, default_flow_style=False, allow_unicode=True)


# CLI helper functions

async def expand_intent(
    intent_text: str,
    target: Optional[str] = None,
    environment: Optional[str] = None,
    hints: Optional[List[str]] = None,
) -> str:
    """
    Quick function to expand an intent to workflow YAML.

    Args:
        intent_text: Natural language intent
        target: Target system/app
        environment: Target environment
        hints: Execution hints

    Returns:
        Executable workflow YAML
    """
    intent = BBXIntent(
        intent=intent_text,
        target=target,
        environment=environment,
        hints=hints or [],
    )

    engine = IntentEngine()
    expanded = await engine.expand(intent)
    return engine.to_executable_yaml(expanded)


async def run_intent(
    intent_or_path: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Expand and optionally run an intent.

    Args:
        intent_or_path: Intent text or path to .bbx file
        dry_run: If True, only expand without running

    Returns:
        Execution result or expanded workflow
    """
    # Determine if it's a file path or intent text
    if Path(intent_or_path).exists():
        intent = BBXIntent.from_file(intent_or_path)
    else:
        intent = BBXIntent(intent=intent_or_path)

    engine = IntentEngine()
    expanded = await engine.expand(intent)

    if dry_run:
        return {
            "status": "dry_run",
            "intent": intent.intent,
            "confidence": expanded.confidence,
            "sources": expanded.sources,
            "steps": expanded.steps,
            "warnings": expanded.warnings,
            "yaml": engine.to_executable_yaml(expanded),
        }

    # TODO: Actually run the workflow
    # from blackbox.core.runtime import run_workflow
    # return await run_workflow(expanded.steps)

    return {
        "status": "expanded",
        "message": "Workflow expanded. Use bbx run <workflow.yaml> to execute.",
        "yaml": engine.to_executable_yaml(expanded),
    }
