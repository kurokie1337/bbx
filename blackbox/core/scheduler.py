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

from pathlib import Path
from typing import Any, Dict, List

import yaml


class Scheduler:
    """
    Scans for workflows with 'triggers' definition.
    """

    @staticmethod
    def scan_triggers(directory: str = ".") -> List[Dict[str, Any]]:
        triggers = []
        root_dir = Path(directory)

        for file_path in root_dir.rglob("*.bbx"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data or "triggers" not in data:
                    continue

                wf_triggers = data["triggers"]
                if not isinstance(wf_triggers, list):
                    continue

                for trigger in wf_triggers:
                    triggers.append(
                        {
                            "workflow": str(file_path),
                            "workflow_id": data.get("id", "unknown"),
                            "type": trigger.get("type"),
                            "schedule": trigger.get("schedule"),
                            "details": trigger,
                        }
                    )
            except Exception:
                continue

        return triggers
