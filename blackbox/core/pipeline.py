# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
Container Pipeline Orchestration
Chains multiple containers in sequence or parallel
"""

import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PipelineStep:
    """Single step in a container pipeline."""
    name: str
    image: str
    command: List[str]
    env: Dict[str, str]
    depends_on: List[str]  # Step names this depends on
    parallel: bool = False

class ContainerPipeline:
    """
    Orchestrates multi-container pipelines.
    
    Features:
    - Sequential execution
    - Parallel execution
    - Dependency resolution
    - Data passing between containers
    """
    
    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize pipeline.
        
        Args:
            steps: List of step definitions
        """
        self.logger = logging.getLogger("bbx.pipeline")
        self.steps = self._parse_steps(steps)
        self.results: Dict[str, Any] = {}
    
    def _parse_steps(self, steps: List[Dict]) -> List[PipelineStep]:
        """Parse step definitions."""
        parsed = []
        for step_def in steps:
            parsed.append(PipelineStep(
                name=step_def["name"],
                image=step_def["image"],
                command=step_def["command"],
                env=step_def.get("env", {}),
                depends_on=step_def.get("depends_on", []),
                parallel=step_def.get("parallel", False)
            ))
        return parsed
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the entire pipeline."""
        self.logger.info(f"🚀 Executing pipeline with {len(self.steps)} steps")
        
        # Build dependency graph
        graph = self._build_dependency_graph()
        
        # Execute in topological order
        levels = self._get_execution_levels(graph)
        
        for level_idx, level in enumerate(levels):
            self.logger.info(f"📍 Level {level_idx + 1}: {len(level)} step(s)")
            
            # Execute level steps in parallel
            tasks = []
            for step_name in level:
                step = self._get_step(step_name)
                task = self._execute_step(step)
                tasks.append(task)
            
            # Wait for level to complete
            await asyncio.gather(*tasks)
        
        return {
            "success": True,
            "steps_executed": len(self.results),
            "results": self.results
        }
    
    async def _execute_step(self, step: PipelineStep) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        self.logger.info(f"▶️  Executing: {step.name}")
        
        # Prepare inputs with previous results
        inputs = {
            "uses": step.image,
            "cmd": step.command,
            "env": step.env
        }
        
        # Add results from dependencies
        for dep in step.depends_on:
            if dep in self.results:
                inputs[f"dep_{dep}"] = self.results[dep]
        
        # Execute
        adapter = UniversalAdapterV2()
        result = await adapter.execute("run", inputs)
        
        # Store result
        self.results[step.name] = result
        
        if result.get("success"):
            self.logger.info(f"✅ {step.name} completed")
        else:
            self.logger.error(f"❌ {step.name} failed: {result.get('error')}")
        
        return result
    
    def _get_step(self, name: str) -> PipelineStep:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        raise ValueError(f"Step not found: {name}")
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph."""
        graph = {}
        for step in self.steps:
            graph[step.name] = step.depends_on
        return graph
    
    def _get_execution_levels(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Get execution levels using topological sort."""
        in_degree = {name: len(deps) for name, deps in graph.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        
        levels = []
        
        while queue:
            # Current level
            current_level = list(queue)
            levels.append(current_level)
            queue.clear()
            
            # Process level
            for node in current_level:
                # Reduce in-degree for dependents
                for name, deps in graph.items():
                    if node in deps:
                        in_degree[name] -= 1
                        if in_degree[name] == 0:
                            queue.append(name)
        
        return levels
