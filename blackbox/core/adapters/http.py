import httpx
from typing import Dict, Any
from blackbox.core.base_adapter import MCPAdapter

class LocalHttpAdapter(MCPAdapter):
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        url = inputs.get("url")
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
            except Exception as e:
                content = response.text

            return {
                "status_code": response.status_code,
                "body": content,
                "headers": dict(response.headers)
            }
