"""
Telegram MCP Adapter for Blackbox
Send messages via Telegram Bot API
"""

import httpx
from typing import Dict, Any


class TelegramAdapter:
    """Telegram Bot API adapter"""
    
    def __init__(self, bot_token: str = None):
        """
        Initialize Telegram adapter
        
        Args:
            bot_token: Telegram bot token (or pass in inputs)
        """
        self.bot_token = bot_token
        self.client = httpx.AsyncClient()
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute Telegram method"""
        
        # Get bot token from inputs or instance
        bot_token = inputs.get("bot_token", self.bot_token)
        if not bot_token:
            raise ValueError("bot_token is required")
        
        if method == "send_message" or method == "send":
            return await self._send_message(bot_token, inputs)
        
        else:
            raise ValueError(f"Unknown Telegram method: {method}")
    
    async def _send_message(self, bot_token: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message via Telegram"""
        chat_id = inputs.get("chat_id")
        text = inputs.get("text") or inputs.get("message", "")
        
        if not chat_id:
            raise ValueError("chat_id is required")
        
        if not text:
            raise ValueError("text/message is required")
        
        # Telegram Bot API endpoint
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Send message
        response = await self.client.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": inputs.get("parse_mode", "HTML")
        })
        
        response.raise_for_status()
        result = response.json()
        
        return {
            "ok": result.get("ok"),
            "message_id": result.get("result", {}).get("message_id"),
            "chat_id": chat_id,
            "text": text
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
