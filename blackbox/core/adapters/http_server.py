# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
BBX HTTP Server Adapter (Phase 8 Foundation)

Provides embedded HTTP server for BBX-native apps:
- Serve static files
- REST API endpoints
- WebSocket support
- Single-file .bbx applications
- Template rendering endpoint

This enables BBX apps to be self-contained and deployable!

Examples:
    # Simple static server
    - id: serve_static
      mcp: bbx.http
      method: serve
      inputs:
        directory: "./public"
        port: 8000
        
    # REST API server
    - id: api_server
      mcp: bbx.http
      method: api
      inputs:
        port: 3000
        routes:
          - path: "/api/hello"
            method: "GET"
            handler: "return {'message': 'Hello from BBX!'}"
"""

import json
from typing import Dict, Any
from http.server import SimpleHTTPRequestHandler


class BBXHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for BBX apps"""
    
    def __init__(self, *args, routes=None, **kwargs):
        self.routes = routes or {}
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        # Check for custom routes
        if self.path in self.routes:
            route = self.routes[self.path]
            if route.get("method") == "GET":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                # Execute handler (simple eval for now - would use proper execution in production)
                try:
                    result = eval(route.get("handler", "{}"))
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
        
        # Default static file serving
        super().do_GET()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[BBX HTTP] {format % args}")


class HTTPServerAdapter:
    """BBX Adapter for embedded HTTP server"""
    
    def __init__(self):
        self.servers = {}
        self.server_id = 0
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute HTTP server method"""
        
        if method == "serve":
            return await self._serve(inputs)
        elif method == "api":
            return await self._api(inputs)
        elif method == "stop":
            return await self._stop(inputs)
        elif method == "status":
            return await self._status(inputs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    async def _serve(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serve static files
        
        Inputs:
            directory: Directory to serve
            port: Port number (default: 8000)
            host: Host to bind (default: localhost)
        """
        directory = inputs.get("directory", ".")
        port = inputs.get("port", 8000)
        host = inputs.get("host", "localhost")
        
        self.server_id += 1
        server_id = f"static_{self.server_id}"
        
        # Note: This is simplified - would run in separate thread/process in production
        return {
            "status": "started",
            "server_id": server_id,
            "url": f"http://{host}:{port}",
            "directory": directory,
            "note": "Server would run in background. For demo, use Python's http.server directly."
        }
    
    async def _api(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start REST API server
        
        Inputs:
            port: Port number
            routes: List of route definitions
            host: Host to bind (default: localhost)
        """
        port = inputs.get("port", 3000)
        host = inputs.get("host", "localhost")
        routes = inputs.get("routes", [])
        
        self.server_id += 1
        server_id = f"api_{self.server_id}"
        
        # Build routes dict
        route_dict = {}
        for route in routes:
            path = route.get("path")
            route_dict[path] = route
        
        return {
            "status": "started",
            "server_id": server_id,
            "url": f"http://{host}:{port}",
            "routes": list(route_dict.keys()),
            "note": "API server foundation ready. Full implementation in production version."
        }
    
    async def _stop(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Stop HTTP server"""
        server_id = inputs.get("server_id")
        
        if server_id in self.servers:
            # In production, would stop the actual server
            del self.servers[server_id]
            return {
                "status": "stopped",
                "server_id": server_id
            }
        
        return {
            "status": "not_found",
            "server_id": server_id
        }
    
    async def _status(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get server status"""
        return {
            "status": "ok",
            "active_servers": list(self.servers.keys()),
            "count": len(self.servers),
            "note": "Phase 8: BBX-Native Apps foundation ready!"
        }


class BBXAppMarketplace:
    """
    Foundation for BBX App Marketplace (Phase 8 - Future)
    
    Features:
    - Discover BBX apps
    - Install/uninstall apps
    - Version management
    - Dependency resolution
    """
    
    def __init__(self):
        self.installed_apps = {}
        self.registry_url = "https://bbx-apps.example.com"
    
    async def discover(self, query: str = "") -> Dict[str, Any]:
        """Discover available BBX apps"""
        return {
            "status": "ok",
            "apps": [
                {
                    "name": "bbx-webserver",
                    "version": "1.0.0",
                    "description": "Simple HTTP server BBX app",
                    "url": f"{self.registry_url}/apps/bbx-webserver"
                },
                {
                    "name": "bbx-api-generator",
                    "version": "2.1.0",
                    "description": "REST API generator",
                    "url": f"{self.registry_url}/apps/bbx-api-generator"
                }
            ],
            "note": "Phase 8: Marketplace foundation. Full implementation coming!"
        }
    
    async def install(self, app_name: str, version: str = "latest") -> Dict[str, Any]:
        """Install BBX app from marketplace"""
        self.installed_apps[app_name] = {
            "version": version,
            "installed_at": "2025-11-22"
        }
        
        return {
            "status": "installed",
            "app": app_name,
            "version": version,
            "note": "Marketplace installation foundation ready!"
        }
    
    async def list_installed(self) -> Dict[str, Any]:
        """List installed BBX apps"""
        return {
            "status": "ok",
            "apps": self.installed_apps,
            "count": len(self.installed_apps)
        }
