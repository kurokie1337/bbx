
import logging
import subprocess
import time
import json
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional

from blackbox.core.base_adapter import BaseAdapter, AdapterResponse, AdapterErrorType

class SearxAdapter(BaseAdapter):
    """
    Sovereign Search Adapter using local SearXNG container.
    
    Provides meta-search capabilities (Google, Bing, etc.) without API keys:
    1. Manages a local `bbx-searxng` Docker container.
    2. Proxies search queries to the local instance.
    3. Returns clean, consistent results.
    """
    
    CONTAINER_NAME = "bbx-searxng"
    IMAGE_NAME = "searxng/searxng:latest"
    PORT = 8080
    
    def __init__(self):
        super().__init__("SearxAdapter")
        self.register_method("search", self.search)
        self.register_method("ensure_server", self.ensure_server)
        self.register_method("stop_server", self.stop_server)
        self._server_ready = False

    async def search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a search query.
        
        Args:
            query (str): Search query
            categories (str): Comma-separated categories (general, science, it, etc.)
            limit (int): Number of results to return
            
        Returns:
            Dict with 'results' list and 'suggestions'
        """
        query = inputs.get("query")
        if not query:
            return AdapterResponse.error_response("Missing 'query' parameter").to_dict()
            
        categories = inputs.get("categories", "general")
        limit = inputs.get("limit", 10)
        
        # Ensure server is running
        if not self._check_server_running():
            self._start_server()
            
        # Build URL
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        url = f"http://localhost:{self.PORT}/search?{urllib.parse.urlencode(params)}"
        
        try:
            # Execute request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    return AdapterResponse.error_response(
                        f"SearXNG returned status {response.status}"
                    ).to_dict()
                    
                data = json.loads(response.read().decode("utf-8"))
                
                # Parse and filter results
                results = []
                for res in data.get("results", [])[:limit]:
                    results.append({
                        "title": res.get("title"),
                        "url": res.get("url"),
                        "content": res.get("content"),
                        "engine": res.get("engine"),
                        "score": res.get("score")
                    })
                    
                return AdapterResponse.success_response({
                    "results": results,
                    "suggestions": data.get("suggestions", []),
                    "answers": data.get("answers", [])
                }).to_dict()
                
        except Exception as e:
            return AdapterResponse.error_response(
                f"Search failed: {str(e)}", 
                error_type=AdapterErrorType.EXECUTION_ERROR
            ).to_dict()

    async def ensure_server(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure SearXNG server is running."""
        running = self._check_server_running()
        if not running:
            success = self._start_server()
            return AdapterResponse.success_response({
                "running": success, 
                "message": "Server started" if success else "Failed to start"
            }).to_dict()
        return AdapterResponse.success_response({"running": True, "message": "Already running"}).to_dict()

    async def stop_server(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the SearXNG container."""
        try:
            subprocess.run(["docker", "rm", "-f", self.CONTAINER_NAME], 
                         check=False, capture_output=True)
            self._server_ready = False
            return AdapterResponse.success_response("Server stopped").to_dict()
        except Exception as e:
            return AdapterResponse.error_response(str(e)).to_dict()

    def _check_server_running(self) -> bool:
        """Check if container is running and responsive."""
        # 1. Check Docker container status
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.CONTAINER_NAME],
            capture_output=True, text=True
        )
        if res.returncode != 0 or res.stdout.strip() != "true":
            return False
            
        # 2. Check HTTP health
        try:
            req = urllib.request.Request(f"http://localhost:{self.PORT}/healthz")
            with urllib.request.urlopen(req, timeout=1) as response:
                return response.status == 200
        except Exception:
            return False

    def _start_server(self) -> bool:
        """Start the SearXNG container."""
        self.logger.info("Starting SearXNG container...")
        
        # 1. Remove existing dead container
        subprocess.run(["docker", "rm", "-f", self.CONTAINER_NAME], 
                     capture_output=True)
                     
        # 2. Run new container
        # Note: minimal config, disabling limiter for API usage
        cmd = [
            "docker", "run", "-d",
            "--name", self.CONTAINER_NAME,
            "-p", f"{self.PORT}:8080",
            "-e", "SEARXNG_BASE_URL=http://localhost:8080/",
            "-e", "SEARXNG_API_ENABLED=true",
            # Disable rate limiting for local usage
            "-e", "SEARXNG_LIMITER=false", 
            self.IMAGE_NAME
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                self.logger.error(f"Failed to start docker: {res.stderr}")
                return False
                
            # 3. Wait for health check (up to 10s)
            for _ in range(20):
                time.sleep(0.5)
                if self._check_server_running():
                    self.logger.info("SearXNG is ready!")
                    self._server_ready = True
                    return True
            
            self.logger.error("SearXNG timed out waiting for health check")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting SearXNG: {e}")
            return False
