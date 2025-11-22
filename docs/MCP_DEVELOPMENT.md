# MCP Adapter Development Guide

> **Complete guide to creating custom MCP adapters for Blackbox**

## 🎯 What is an MCP Adapter?

MCP (Model Context Protocol) adapters are **plugins** that extend Blackbox functionality. Each adapter represents a category of operations (HTTP, SQL, file I/O, etc.).

**Think of adapters as:**
- API clients
- Service wrappers
- Integration points
- Reusable components

---

## 🏗️ Adapter Architecture

```
BBX Workflow
    ↓
Runtime Engine
    ↓
MCP Registry ───→ HTTP Adapter ───→ External API
              ├──→ SQL Adapter  ───→ Database
              ├──→ File Adapter ───→ Filesystem
              └──→ Custom Adapter → Your Service
```

---

## 📝 Basic Adapter Template

### Minimal Implementation

```python
from typing import Dict, Any

class MCPAdapter:
    """Base interface for all MCP adapters"""
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute an adapter method.
        
        Args:
            method: Method name to invoke
            inputs: Method arguments as key-value pairs
            
        Returns:
            Serializable result (dict, list, str, int, bool, None)
        """
        raise NotImplementedError
```

### Simple Example: Logger Adapter

```python
import logging
from typing import Dict, Any
from blackbox.core.adapters.base import MCPAdapter

class LoggerAdapter(MCPAdapter):
    """Simple logging adapter"""
    
    def __init__(self):
        self.logger = logging.getLogger("blackbox")
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        message = inputs.get("message", "")
        
        if method == "info":
            self.logger.info(message)
            return {"status": "logged", "level": "info"}
        
        elif method == "error":
            self.logger.error(message)
            return {"status": "logged", "level": "error"}
        
        elif method == "warning":
            self.logger.warning(message)
            return {"status": "logged", "level": "warning"}
        
        else:
            raise ValueError(f"Unknown method: {method}")
```

**Usage in BBX:**

```yaml
steps:
  - id: "log_message"
    mcp: "logger"
    method: "info"
    inputs:
      message: "Workflow started at ${env.TIMESTAMP}"
```

---

## 🚀 Advanced Adapter: HTTP Client

### Full Implementation

```python
import httpx
from typing import Dict, Any, Optional
from blackbox.core.adapters.base import MCPAdapter

class HTTPAdapter(MCPAdapter):
    """HTTP client adapter with connection pooling"""
    
    def __init__(self, timeout: int = 30, max_connections: int = 100):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_connections)
        )
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute HTTP request"""
        url = inputs.get("url")
        if not url:
            raise ValueError("Missing required parameter: url")
        
        headers = inputs.get("headers", {})
        params = inputs.get("params", {})
        
        if method == "get":
            return await self._get(url, headers, params)
        elif method == "post":
            return await self._post(url, headers, inputs.get("json"), inputs.get("data"))
        elif method == "put":
            return await self._put(url, headers, inputs.get("json"))
        elif method == "delete":
            return await self._delete(url, headers)
        else:
            raise ValueError(f"Unknown HTTP method: {method}")
    
    async def _get(self, url: str, headers: dict, params: dict) -> dict:
        """Execute GET request"""
        response = await self.client.get(url, headers=headers, params=params)
        return self._format_response(response)
    
    async def _post(self, url: str, headers: dict, json_data: Optional[dict], form_data: Optional[dict]) -> dict:
        """Execute POST request"""
        response = await self.client.post(
            url,
            headers=headers,
            json=json_data,
            data=form_data
        )
        return self._format_response(response)
    
    async def _put(self, url: str, headers: dict, json_data: Optional[dict]) -> dict:
        """Execute PUT request"""
        response = await self.client.put(url, headers=headers, json=json_data)
        return self._format_response(response)
    
    async def _delete(self, url: str, headers: dict) -> dict:
        """Execute DELETE request"""
        response = await self.client.delete(url, headers=headers)
        return self._format_response(response)
    
    def _format_response(self, response: httpx.Response) -> dict:
        """Format HTTP response"""
        content_type = response.headers.get("content-type", "")
        
        # Parse JSON responses
        if "application/json" in content_type:
            try:
                data = response.json()
            except Exception:
                data = response.text
        else:
            data = response.text
        
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "data": data,
            "ok": response.status_code < 400
        }
    
    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()
```

---

## 🎨 Design Patterns

### Pattern 1: Stateful Adapter

Maintain connection state across multiple calls:

```python
class DatabaseAdapter(MCPAdapter):
    def __init__(self, connection_string: str):
        self.connection = None
        self.connection_string = connection_string
    
    async def _ensure_connected(self):
        """Lazy connection initialization"""
        if self.connection is None:
            self.connection = await asyncpg.connect(self.connection_string)
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        await self._ensure_connected()
        
        if method == "query":
            query = inputs.get("sql")
            return await self.connection.fetch(query)
        # ...
```

### Pattern 2: Configuration-Based Adapter

Load settings from environment or config:

```python
class TelegramAdapter(MCPAdapter):
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        
        self.client = telegram.Bot(token=self.token)
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "send_message":
            chat_id = inputs.get("chat_id")
            text = inputs.get("text")
            return await self.client.send_message(chat_id, text)
```

### Pattern 3: Multi-Method Router

Clean method dispatch:

```python
class FileAdapter(MCPAdapter):
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        # Method dispatch table
        methods = {
            "read": self._read_file,
            "write": self._write_file,
            "append": self._append_file,
            "delete": self._delete_file,
            "exists": self._file_exists,
        }
        
        handler = methods.get(method)
        if not handler:
            raise ValueError(f"Unknown method: {method}")
        
        return await handler(inputs)
    
    async def _read_file(self, inputs: Dict[str, Any]) -> str:
        path = inputs.get("path")
        async with aiofiles.open(path, "r") as f:
            return await f.read()
    
    async def _write_file(self, inputs: Dict[str, Any]) -> dict:
        path = inputs.get("path")
        content = inputs.get("content")
        async with aiofiles.open(path, "w") as f:
            await f.write(content)
        return {"status": "written", "path": path}
```

---

## 🔐 Best Practices

### 1. Input Validation

Always validate required parameters:

```python
async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
    # Validate required fields
    url = inputs.get("url")
    if not url:
        raise ValueError("Missing required parameter: url")
    
    # Validate types
    timeout = inputs.get("timeout", 30)
    if not isinstance(timeout, int):
        raise TypeError("timeout must be an integer")
```

### 2. Error Handling

Wrap external calls in try/except:

```python
async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
    try:
        result = await self._make_api_call(inputs)
        return {"status": "success", "data": result}
    except httpx.TimeoutException as e:
        raise Exception(f"Request timeout: {e}")
    except httpx.HTTPError as e:
        raise Exception(f"HTTP error: {e}")
```

### 3. Resource Cleanup

Implement cleanup method:

```python
class MyAdapter(MCPAdapter):
    async def close(self):
        """Cleanup resources on shutdown"""
        if self.client:
            await self.client.aclose()
        if self.connection:
            await self.connection.close()
```

### 4. Logging

Add debug logging:

```python
import logging

class MyAdapter(MCPAdapter):
    def __init__(self):
        self.logger = logging.getLogger(f"blackbox.adapter.{self.__class__.__name__}")
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        self.logger.debug(f"Executing {method} with inputs: {inputs}")
        result = await self._do_work(inputs)
        self.logger.debug(f"Result: {result}")
        return result
```

---

## 📦 Adapter Registration

### Manual Registration

```python
from blackbox.core.registry import MCPRegistry
from myapp.adapters import TelegramAdapter, SQLAdapter

# Create registry
registry = MCPRegistry()

# Register adapters
registry.register("telegram", TelegramAdapter())
registry.register("sql", SQLAdapter("postgresql://localhost/mydb"))

# Use in runtime
from blackbox.core import run_file
result = await run_file("workflow.bbx", registry=registry)
```

### Auto-Discovery (Advanced)

```python
import importlib
import inspect
from pathlib import Path

def auto_register_adapters(registry: MCPRegistry, package_path: str):
    """Auto-discover and register adapters"""
    adapter_dir = Path(package_path)
    
    for file in adapter_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        
        # Import module
        module = importlib.import_module(f"adapters.{file.stem}")
        
        # Find adapter classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, MCPAdapter) and obj != MCPAdapter:
                adapter_name = name.lower().replace("adapter", "")
                registry.register(adapter_name, obj())
```

---

## 🧪 Testing Adapters

### Unit Test Template

```python
import pytest
from myapp.adapters import HTTPAdapter

@pytest.fixture
async def adapter():
    adapter = HTTPAdapter()
    yield adapter
    await adapter.close()

@pytest.mark.asyncio
async def test_get_request(adapter):
    result = await adapter.execute("get", {
        "url": "https://httpbin.org/get"
    })
    
    assert result["status_code"] == 200
    assert "data" in result

@pytest.mark.asyncio
async def test_missing_url_raises_error(adapter):
    with pytest.raises(ValueError, match="Missing required parameter: url"):
        await adapter.execute("get", {})
```

---

## 📚 Real-World Examples

### Slack Adapter

```python
from slack_sdk.web.async_client import AsyncWebClient

class SlackAdapter(MCPAdapter):
    def __init__(self):
        self.client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "send_message":
            channel = inputs.get("channel")
            text = inputs.get("text")
            response = await self.client.chat_postMessage(
                channel=channel,
                text=text
            )
            return {"ok": response["ok"], "ts": response["ts"]}
```

### AWS S3 Adapter

```python
import aioboto3

class S3Adapter(MCPAdapter):
    def __init__(self):
        self.session = aioboto3.Session()
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        async with self.session.client("s3") as s3:
            if method == "upload":
                bucket = inputs.get("bucket")
                key = inputs.get("key")
                data = inputs.get("data")
                await s3.put_object(Bucket=bucket, Key=key, Body=data)
                return {"bucket": bucket, "key": key, "status": "uploaded"}
```

---

## 🔮 Advanced Topics

### Streaming Responses

```python
async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
    if method == "stream":
        url = inputs.get("url")
        async with self.client.stream("GET", url) as response:
            chunks = []
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
            return b"".join(chunks)
```

### Rate Limiting

```python
from asyncio import Semaphore

class RateLimitedAdapter(MCPAdapter):
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = Semaphore(max_concurrent)
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        async with self.semaphore:
            return await self._do_work(inputs)
```

---

## 📖 See Also

- **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Complete workflow format reference
- **[Universal Adapter Guide](UNIVERSAL_ADAPTER.md)** - Zero-code adapter architecture
- **[Runtime Internals](RUNTIME_INTERNALS.md)** - Engine implementation details
- **[Architecture Guide](ARCHITECTURE.md)** - System design
- **[Getting Started](GETTING_STARTED.md)** - Beginner's guide
- **[Documentation Index](INDEX.md)** - Complete documentation navigation

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0
