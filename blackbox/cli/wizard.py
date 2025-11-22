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

import os

import click
import yaml


class WorkflowWizard:
    """
    Interactive wizard for generating Blackbox workflows.
    """

    TEMPLATES = {
        "empty": {
            "name": "Empty Workflow",
            "description": "Start from scratch",
            "steps": {},
            "params": [],
        },
        "scraper": {
            "name": "Web Scraper",
            "description": "Scrape a website content and save to file",
            "steps": {
                "open_page": {"use": "browser.open", "args": {"url": "${url}"}},
                "get_content": {"use": "browser.text", "args": {"selector": "body"}},
                "save_file": {
                    "use": "system.shell",
                    "args": {
                        "command": "echo '${get_content.output}' > ${output_file}"
                    },
                },
            },
            "params": ["url", "output_file"],
        },
        "api_monitor": {
            "name": "API Monitor",
            "description": "Check API health and log result",
            "steps": {
                "check_api": {"use": "http.get", "args": {"url": "${url}"}},
                "log_success": {
                    "use": "logger.info",
                    "args": {"message": "API is UP: ${check_api.output}"},
                    "when": "${check_api.status} == 'success'",
                },
                "log_failure": {
                    "use": "logger.error",
                    "args": {"message": "API is DOWN"},
                    "when": "${check_api.status} != 'success'",
                },
            },
            "params": ["url"],
        },
        "telegram_bot": {
            "name": "Telegram Notifier",
            "description": "Send a message to Telegram",
            "steps": {
                "send_msg": {
                    "use": "telegram.send",
                    "args": {"chat_id": "${chat_id}", "text": "${message}"},
                }
            },
            "params": ["chat_id", "message"],
        },
    }

    @staticmethod
    def run():
        click.echo("🧙 Welcome to Blackbox Workflow Wizard!")
        click.echo("Let's create a new workflow.\n")

        # 1. Basic Info
        name = click.prompt("Workflow Name", default="My Workflow")
        wf_id = click.prompt("Workflow ID", default=name.lower().replace(" ", "_"))
        filename = click.prompt("Filename", default=f"{wf_id}.bbx")

        if not filename.endswith(".bbx"):
            filename += ".bbx"

        if os.path.exists(filename):
            if not click.confirm(f"File {filename} exists. Overwrite?", default=False):
                click.echo("Aborted.")
                return

        # 2. Choose Template
        click.echo("\nAvailable Templates:")
        template_keys = list(WorkflowWizard.TEMPLATES.keys())
        for i, key in enumerate(template_keys):
            tmpl = WorkflowWizard.TEMPLATES[key]
            click.echo(f"  {i+1}. {tmpl['name']} - {tmpl['description']}")

        choice = click.prompt("\nChoose a template", type=int, default=1)
        if choice < 1 or choice > len(template_keys):
            click.echo("Invalid choice. Defaulting to Empty.")
            selected_key = "empty"
        else:
            selected_key = template_keys[choice - 1]

        template = WorkflowWizard.TEMPLATES[selected_key]
        click.echo(f"\nSelected: {template['name']}")

        # 3. Configure Parameters
        steps = template["steps"].copy()

        # We need to replace placeholders in steps with user values
        # Or we can just set them as default inputs?
        # Let's replace them in the YAML for simplicity, so the file is ready to run.

        replacements = {}
        if template["params"]:
            click.echo("\nConfiguration:")
            for param in template["params"]:
                val = click.prompt(f"  Value for '{param}'")
                replacements[f"${{{param}}}"] = val

        # Recursively replace in steps
        def replace_recursive(data):
            if isinstance(data, dict):
                return {k: replace_recursive(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [replace_recursive(i) for i in data]
            elif isinstance(data, str):
                for k, v in replacements.items():
                    data = data.replace(k, v)
                return data
            else:
                return data

        final_steps = replace_recursive(steps)

        # 4. Generate YAML
        workflow_data = {
            "id": wf_id,
            "name": name,
            "version": "1.0.0",
            "steps": final_steps,
        }

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(workflow_data, f, sort_keys=False, allow_unicode=True)

        click.echo(f"\n✨ Workflow created successfully: {filename}")
        click.echo(f"Run it with: blackbox run {filename}")
