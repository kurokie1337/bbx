
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from blackbox.core.base_adapter import DockerizedAdapter, AdapterResponse, AdapterErrorType

class SimpleHTMLTextExtractor(HTMLParser):
    """
    Zero-dependency HTML to Text converter using standard library.
    Extracts readable text from HTML, ignoring scripts and styles.
    """
    def __init__(self):
        super().__init__()
        self.result = []
        self.in_script = False
        
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style', 'head', 'title', 'meta', 'link']:
            self.in_script = True
        elif tag in ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']:
            self.result.append('\n')
            
    def handle_endtag(self, tag):
        if tag in ['script', 'style', 'head', 'title', 'meta', 'link']:
            self.in_script = False
        elif tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']:
            self.result.append('\n')

    def handle_data(self, data):
        if not self.in_script and data.strip():
            self.result.append(data.strip() + ' ')
            
    def get_text(self):
        text = "".join(self.result)
        # Clean up excessive newlines
        return re.sub(r'\n\s*\n', '\n\n', text).strip()

class BrowserAdapter(DockerizedAdapter):
    """
    Headless Browser Adapter for Sovereign Research.
    
    Uses `zenika/alpine-chrome` to fetch web pages with JavaScript execution capabilities,
    rendering the DOM structure which is then parsed into clean text.
    """
    
    def __init__(self):
        super().__init__(
            adapter_name="BrowserAdapter",
            docker_image="zenika/alpine-chrome:latest"
        )
        self.register_method("visit", self.visit)
        self.register_method("extract_text", self.extract_text)
        self.register_method("screenshot", self.screenshot)

    async def visit(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visit a URL and return the rendered HTML.
        
        Args:
            url (str): URL to visit
            wait (int): Time to wait for JS execution (default: 2s)
            
        Returns:
            Dict with 'html', 'url'
        """
        url = inputs.get("url")
        if not url:
            return AdapterResponse.error_response("Missing 'url' parameter").to_dict()
            
        wait = inputs.get("wait", 2)
        
        # We use --dump-dom to get the rendered state
        # The image entrypoint is 'chromium-browser'
        # Arguments: --headless --dump-dom <url>
        
        # We need to rely on the entrypoint of the image or force it.
        # zenika/alpine-chrome entrypoint is `chromium-browser`
        # Default run_command appends args to entrypoint.
        
        flags = [
            "--headless",
            "--disable-gpu",
            "--no-sandbox", # Often needed in Docker
            "--dump-dom",
            # Improve stealth/reliability
            "--disable-dev-shm-usage",
            "--window-size=1920,1080",
            f"--virtual-time-budget={wait*1000}", # waiting for JS
            url
        ]
        
        result = self.run_command(*flags)
        
        if not result.success:
            return result.to_dict()
            
        return AdapterResponse.success_response({
            "url": url,
            "html": result.data.get("stdout", "")
        }).to_dict()

    async def extract_text(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visit a URL and extract readable text content.
        
        Args:
            url (str): URL to visit
            
        Returns:
            Dict with 'text', 'title'
        """
        # 1. Get HTML
        response = await self.visit(inputs)
        if not response.get("success"):
            return response
            
        html_content = response["data"].get("html", "")
        
        # 2. Parse Text
        try:
            parser = SimpleHTMLTextExtractor()
            parser.feed(html_content)
            text = parser.get_text()
            
            return AdapterResponse.success_response({
                "url": inputs["url"],
                "text": text,
                "length": len(text)
            }).to_dict()
            
        except Exception as e:
            return AdapterResponse.error_response(
                f"Text extraction failed: {e}",
                error_type=AdapterErrorType.EXECUTION_ERROR
            ).to_dict()

    async def screenshot(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take a screenshot of a page.
        (Note: Requires volume mounting to save file, returning binary not supported efficiently via CLI stdout)
        """
        return AdapterResponse.error_response("Screenshot not yet implemented for CLI mode").to_dict()
