# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BBX Format v6.0 Parser

Simplified BBX syntax with:
- Step IDs as YAML keys (not in 'id' field)
- Combined 'use' field (adapter.method)
- Shorter field names ('args' instead of 'inputs', 'save' instead of 'outputs')
- Human-readable durations ('5s', '1m', '10ms')
"""

import yaml
import re
from typing import Dict, Any


class BBXv6Parser:
    """Parser for BBX Format v6.0"""
    
    @staticmethod
    def parse_duration(duration_str: str) -> int:
        """
        Parse human-readable duration to milliseconds.
        
        Examples:
            '5s' -> 5000
            '1m' -> 60000
            '500ms' -> 500
            '30' -> 30000 (defaults to ms if no unit)
        
        Args:
            duration_str: Duration string
            
        Returns:
            Duration in milliseconds
        """
        if isinstance(duration_str, (int, float)):
            return int(duration_str)
        
        duration_str = str(duration_str).strip()
        
        # Parse with regex
        match = re.match(r'^(\d+(?:\.\d+)?)(ms|s|m|h)?$', duration_str)
        if not match:
            raise ValueError(f"Invalid duration format: {duration_str}")
        
        value = float(match.group(1))
        unit = match.group(2) or 'ms'  # Default to milliseconds
        
        # Convert to milliseconds
        multipliers = {
            'ms': 1,
            's': 1000,
            'm': 60000,
            'h': 3600000
        }
        
        return int(value * multipliers[unit])
    
    @staticmethod
    def parse_step(step_id: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v6.0 step to v5.0 format.
        
        Args:
            step_id: Step identifier
            step_data: v6.0 step data
            
        Returns:
            v5.0 step dictionary
        """
        v5_step: Dict[str, Any] = {
            "id": step_id
        }

        # Parse 'use' field (adapter.method)
        if "use" in step_data:
            parts = step_data["use"].split(".", 1)
            if len(parts) == 2:
                v5_step["mcp"] = parts[0]
                v5_step["method"] = parts[1]
            else:
                raise ValueError(f"Invalid 'use' format: {step_data['use']}. Expected 'adapter.method'")
        else:
            # Fallback to explicit mcp/method
            mcp_value = step_data.get("mcp")
            method_value = step_data.get("method")
            if mcp_value:
                v5_step["mcp"] = mcp_value
            if method_value:
                v5_step["method"] = method_value
        
        # Convert field names
        field_mapping = {
            "args": "inputs",
            "save": "outputs",
            "when": "when",
            "parallel": "parallel",
            "depends_on": "depends_on"
        }
        
        for v6_name, v5_name in field_mapping.items():
            if v6_name in step_data:
                v5_step[v5_name] = step_data[v6_name]
        
        # Parse durations
        if "timeout" in step_data:
            v5_step["timeout"] = BBXv6Parser.parse_duration(step_data["timeout"])
        
        if "retry_delay" in step_data:
            v5_step["retry_delay"] = BBXv6Parser.parse_duration(step_data["retry_delay"])
        
        # Copy numeric fields
        if "retry" in step_data:
            v5_step["retry"] = step_data["retry"]
        
        if "retry_backoff" in step_data:
            v5_step["retry_backoff"] = step_data["retry_backoff"]
        
        # Cache configuration
        if "cache" in step_data:
            cache_config = step_data["cache"]
            if isinstance(cache_config, dict):
                # Convert TTL if needed
                if "ttl" in cache_config:
                    cache_config["ttl"] = BBXv6Parser.parse_duration(cache_config["ttl"])
            v5_step["cache"] = cache_config
        
        return v5_step
    
    @staticmethod
    def parse_v6(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v6.0 BBX format to v5.0.

        Args:
            data: v6.0 workflow data

        Returns:
            v5.0 format data
        """
        workflow: Dict[str, Any] = {}

        v5_data: Dict[str, Any] = {
            "bbx_version": "5.0",
            "type": "workflow",
            "workflow": workflow
        }

        # Copy workflow metadata
        if "id" in data:
            workflow["id"] = data["id"]
        if "name" in data:
            workflow["name"] = data["name"]
        if "version" in data:
            workflow["version"] = data["version"]
        if "description" in data:
            workflow["description"] = data["description"]

        # Parse steps (v6 uses dict with step IDs as keys)
        if "steps" in data:
            steps_data = data["steps"]

            if isinstance(steps_data, dict):
                # v6 format: steps are a dictionary
                v5_steps = []
                for step_id, step_data in steps_data.items():
                    v5_step = BBXv6Parser.parse_step(step_id, step_data)
                    v5_steps.append(v5_step)
                workflow["steps"] = v5_steps
            else:
                # v5 format: steps are a list (backward compat)
                workflow["steps"] = steps_data

        return v5_data
    
    @staticmethod
    def detect_version(data: Dict[str, Any]) -> str:
        """
        Detect BBX format version.
        
        Returns:
            Version string ('5.0' or '6.0')
        """
        # Check explicit version
        if "bbx_version" in data:
            version = data["bbx_version"]
            if version.startswith("6"):
                return "6.0"
            return "5.0"
        
        # Heuristic: v6.0 has top-level 'id' and dict-based 'steps'
        if "id" in data and "steps" in data and isinstance(data["steps"], dict):
            return "6.0"
        
        # Heuristic: v5.0 has 'workflow' container
        if "workflow" in data:
            return "5.0"
        
        # Default to v5.0
        return "5.0"
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load and parse BBX file (auto-detects version).
        
        Args:
            file_path: Path to BBX file
            
        Returns:
            v5.0 format data (converted if needed)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        version = BBXv6Parser.detect_version(data)
        
        if version == "6.0":
            return BBXv6Parser.parse_v6(data)
        else:
            return data
