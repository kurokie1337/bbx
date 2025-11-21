import yaml
from typing import List, Dict, Any
from pathlib import Path

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
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    
                if not data or "triggers" not in data:
                    continue
                    
                wf_triggers = data["triggers"]
                if not isinstance(wf_triggers, list):
                    continue
                    
                for trigger in wf_triggers:
                    triggers.append({
                        "workflow": str(file_path),
                        "workflow_id": data.get("id", "unknown"),
                        "type": trigger.get("type"),
                        "schedule": trigger.get("schedule"),
                        "details": trigger
                    })
            except Exception:
                continue
                
        return triggers
