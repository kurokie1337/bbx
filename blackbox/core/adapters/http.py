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

import httpx
from typing import Dict, Any
from blackbox.core.base_adapter import MCPAdapter

class LocalHttpAdapter(MCPAdapter):
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        url = inputs.get("url")
        if not url:
            raise ValueError("URL is required")

        headers = inputs.get("headers", {})
        body = inputs.get("json") or inputs.get("body")

        async with httpx.AsyncClient() as client:
            if method.lower() == "get":
                response = await client.get(url, headers=headers)
            elif method.lower() == "post":
                response = await client.post(url, headers=headers, json=body)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            try:
                content = response.json()
            except Exception:
                content = response.text

            return {
                "status_code": response.status_code,
                "body": content,
                "headers": dict(response.headers)
            }
