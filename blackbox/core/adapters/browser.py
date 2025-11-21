from typing import Dict, Any, Optional
from blackbox.core.base_adapter import MCPAdapter

try:
    from playwright.async_api import async_playwright, Playwright, Browser, Page
except ImportError:
    async_playwright = None

class BrowserAdapter(MCPAdapter):
    """
    Adapter for controlling a web browser using Playwright.
    Maintains state (browser, page) for the duration of the workflow.
    """
    
    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if async_playwright is None:
            raise ImportError("Playwright is not installed. Run: pip install playwright && playwright install")

        if method == "open":
            return await self._open(inputs)
        elif method == "goto":
            return await self._goto(inputs)
        elif method == "click":
            return await self._click(inputs)
        elif method == "type":
            return await self._type(inputs)
        elif method == "text":
            return await self._text(inputs)
        elif method == "screenshot":
            return await self._screenshot(inputs)
        elif method == "close":
            return await self._close(inputs)
        else:
            raise ValueError(f"Unknown browser method: {method}")

    async def _ensure_page(self):
        if not self.page:
            raise RuntimeError("Browser not open. Call browser.open first.")

    async def _open(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        headless = inputs.get("headless", True)
        
        if not self.playwright:
            self.playwright = await async_playwright().start()
            
        if not self.browser:
            self.browser = await self.playwright.chromium.launch(headless=headless)
            
        if not self.page:
            self.page = await self.browser.new_page()
            
        return {"status": "opened", "headless": headless}

    async def _goto(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_page()
        url = inputs.get("url")
        if not url:
            raise ValueError("URL is required")
            
        await self.page.goto(url)
        return {"url": url, "title": await self.page.title()}

    async def _click(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_page()
        selector = inputs.get("selector")
        if not selector:
            raise ValueError("Selector is required")
            
        await self.page.click(selector)
        return {"clicked": selector}

    async def _type(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_page()
        selector = inputs.get("selector")
        text = inputs.get("text")
        if not selector or text is None:
            raise ValueError("Selector and text are required")
            
        await self.page.fill(selector, text)
        return {"typed": text, "into": selector}

    async def _text(self, inputs: Dict[str, Any]) -> str:
        await self._ensure_page()
        selector = inputs.get("selector")
        if not selector:
            raise ValueError("Selector is required")
            
        return await self.page.text_content(selector)

    async def _screenshot(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_page()
        path = inputs.get("path", "screenshot.png")
        await self.page.screenshot(path=path)
        return {"path": path}

    async def _close(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.page:
            await self.page.close()
            self.page = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        return {"status": "closed"}
