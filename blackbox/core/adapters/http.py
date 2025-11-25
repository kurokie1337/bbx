# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

from typing import Any, Dict

import httpx

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
                "headers": dict(response.headers),
            }
