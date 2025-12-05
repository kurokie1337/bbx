# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Real LLM Provider - No Mocks, Real API Calls.

Supports:
- Anthropic Claude (primary)
- OpenAI GPT
- Ollama (local)

Environment Variables:
    ANTHROPIC_API_KEY - Anthropic API key
    OPENAI_API_KEY - OpenAI API key
    OLLAMA_BASE_URL - Ollama base URL (default: http://localhost:11434)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger("bbx.llm")


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Message:
    """Chat message"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMRequest:
    """Request to LLM"""
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    system: Optional[str] = None
    tools: Optional[List[Dict]] = None


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None
    latency_ms: float = 0


class LLMProviderType(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


# =============================================================================
# Base Provider
# =============================================================================


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False
        self._stats = {
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "errors": 0,
            "total_latency_ms": 0,
        }

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider"""
        pass

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a request"""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a completion"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            **self._stats,
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / self._stats["requests"]
                if self._stats["requests"] > 0 else 0
            ),
        }


# =============================================================================
# Anthropic Provider
# =============================================================================


class AnthropicProvider(BaseLLMProvider):
    """
    Real Anthropic Claude provider.

    Uses the official anthropic SDK for API calls.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    async def initialize(self) -> bool:
        """Initialize Anthropic client"""
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return False

        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self._initialized = True
            logger.info("Anthropic provider initialized")
            return True
        except ImportError:
            logger.error("anthropic package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete using Claude"""
        if not self._initialized or not self._client:
            raise RuntimeError("Anthropic provider not initialized")

        start = time.time()

        try:
            # Convert messages
            messages = []
            system_prompt = request.system or ""

            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Make API call
            response = await self._client.messages.create(
                model=request.model or self.DEFAULT_MODEL,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt if system_prompt else None,
                messages=messages,
            )

            latency = (time.time() - start) * 1000

            # Extract content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            # Update stats
            self._stats["requests"] += 1
            self._stats["tokens_in"] += response.usage.input_tokens
            self._stats["tokens_out"] += response.usage.output_tokens
            self._stats["total_latency_ms"] += latency

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=response.stop_reason or "stop",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                latency_ms=latency,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Anthropic error: {e}")
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream completion from Claude"""
        if not self._initialized or not self._client:
            raise RuntimeError("Anthropic provider not initialized")

        try:
            # Convert messages
            messages = []
            system_prompt = request.system or ""

            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Stream API call
            async with self._client.messages.stream(
                model=request.model or self.DEFAULT_MODEL,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt if system_prompt else None,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Anthropic stream error: {e}")
            raise


# =============================================================================
# OpenAI Provider
# =============================================================================


class OpenAIProvider(BaseLLMProvider):
    """
    Real OpenAI GPT provider.

    Uses the official openai SDK for API calls.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set")
            return False

        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info("OpenAI provider initialized")
            return True
        except ImportError:
            logger.error("openai package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete using GPT"""
        if not self._initialized or not self._client:
            raise RuntimeError("OpenAI provider not initialized")

        start = time.time()

        try:
            # Convert messages
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

            # Make API call
            response = await self._client.chat.completions.create(
                model=request.model or self.DEFAULT_MODEL,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency = (time.time() - start) * 1000

            # Extract content
            content = response.choices[0].message.content or ""

            # Update stats
            self._stats["requests"] += 1
            if response.usage:
                self._stats["tokens_in"] += response.usage.prompt_tokens
                self._stats["tokens_out"] += response.usage.completion_tokens
            self._stats["total_latency_ms"] += latency

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                latency_ms=latency,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"OpenAI error: {e}")
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream completion from GPT"""
        if not self._initialized or not self._client:
            raise RuntimeError("OpenAI provider not initialized")

        try:
            # Convert messages
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

            # Stream API call
            stream = await self._client.chat.completions.create(
                model=request.model or self.DEFAULT_MODEL,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"OpenAI stream error: {e}")
            raise


# =============================================================================
# Ollama Provider (Local)
# =============================================================================


class OllamaProvider(BaseLLMProvider):
    """
    Real Ollama provider for local LLMs.

    Connects to local Ollama server via HTTP API.
    """

    DEFAULT_MODEL = "qwen2.5:0.5b"  # Small fast model

    def __init__(self, base_url: Optional[str] = None):
        super().__init__("ollama")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._session = None
        self._available_models = []

    async def initialize(self) -> bool:
        """Initialize Ollama connection"""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()

            # Test connection and get available models
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    self._initialized = True

                    # Auto-select first available model if default not available
                    if self._available_models and self.DEFAULT_MODEL not in self._available_models:
                        self.DEFAULT_MODEL = self._available_models[0]
                        logger.info(f"Using available model: {self.DEFAULT_MODEL}")
                    logger.info("Ollama provider initialized")
                    return True
                else:
                    logger.warning(f"Ollama not available: {resp.status}")
                    await self._session.close()
                    self._session = None
                    return False

        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete using Ollama"""
        if not self._initialized or not self._session:
            raise RuntimeError("Ollama provider not initialized")

        start = time.time()

        try:
            # Build prompt
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n\n"
            prompt += "Assistant: "

            # Make API call
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": request.model or self.DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama error: {resp.status}")

                data = await resp.json()

            latency = (time.time() - start) * 1000

            # Update stats
            self._stats["requests"] += 1
            self._stats["total_latency_ms"] += latency

            return LLMResponse(
                content=data.get("response", ""),
                model=request.model or self.DEFAULT_MODEL,
                finish_reason="stop",
                usage={
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                },
                latency_ms=latency,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Ollama error: {e}")
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama"""
        if not self._initialized or not self._session:
            raise RuntimeError("Ollama provider not initialized")

        try:
            # Build prompt
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n\n"
            prompt += "Assistant: "

            # Stream API call
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": request.model or self.DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            ) as resp:
                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Ollama stream error: {e}")
            raise

    async def shutdown(self):
        """Shutdown Ollama connection"""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False


# =============================================================================
# LLM Manager
# =============================================================================


class LLMManager:
    """
    Manages LLM providers with automatic fallback.

    Priority order:
    1. Anthropic (if API key available)
    2. OpenAI (if API key available)
    3. Ollama (if server running)
    """

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._primary: Optional[BaseLLMProvider] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all available providers"""
        logger.info("Initializing LLM providers...")

        # Try Anthropic
        anthropic = AnthropicProvider()
        if await anthropic.initialize():
            self._providers["anthropic"] = anthropic
            if not self._primary:
                self._primary = anthropic
                logger.info("Primary LLM: Anthropic Claude")

        # Try OpenAI
        openai = OpenAIProvider()
        if await openai.initialize():
            self._providers["openai"] = openai
            if not self._primary:
                self._primary = openai
                logger.info("Primary LLM: OpenAI GPT")

        # Try Ollama
        ollama = OllamaProvider()
        if await ollama.initialize():
            self._providers["ollama"] = ollama
            if not self._primary:
                self._primary = ollama
                logger.info("Primary LLM: Ollama (local)")

        self._initialized = len(self._providers) > 0

        if not self._initialized:
            logger.error("No LLM providers available!")
            logger.error("Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or start Ollama")

        return self._initialized

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Complete a prompt using the best available provider.

        Args:
            prompt: User prompt
            system: System prompt
            model: Specific model to use
            temperature: Sampling temperature
            max_tokens: Max output tokens
            provider: Force specific provider (anthropic, openai, ollama)

        Returns:
            LLMResponse with content and metadata
        """
        if not self._initialized:
            raise RuntimeError("LLM Manager not initialized")

        # Select provider
        if provider and provider in self._providers:
            selected = self._providers[provider]
        elif self._primary:
            selected = self._primary
        else:
            raise RuntimeError("No LLM provider available")

        # Build request
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        # Execute with fallback
        try:
            return await selected.complete(request)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}, trying fallback...")

            # Try other providers
            for name, prov in self._providers.items():
                if prov != selected:
                    try:
                        return await prov.complete(request)
                    except Exception:
                        continue

            raise RuntimeError("All LLM providers failed")

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from LLM"""
        if not self._initialized:
            raise RuntimeError("LLM Manager not initialized")

        # Select provider
        if provider and provider in self._providers:
            selected = self._providers[provider]
        elif self._primary:
            selected = self._primary
        else:
            raise RuntimeError("No LLM provider available")

        # Build request
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        async for chunk in selected.stream(request):
            yield chunk

    def get_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self._providers.keys())

    def get_primary(self) -> Optional[str]:
        """Get primary provider name"""
        return self._primary.name if self._primary else None

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all providers"""
        return {
            name: prov.get_stats()
            for name, prov in self._providers.items()
        }

    async def shutdown(self):
        """Shutdown all providers"""
        for prov in self._providers.values():
            if hasattr(prov, 'shutdown'):
                await prov.shutdown()
        self._providers.clear()
        self._primary = None
        self._initialized = False


# =============================================================================
# Global Instance
# =============================================================================


_llm_manager: Optional[LLMManager] = None


async def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
        await _llm_manager.initialize()
    return _llm_manager


async def complete(
    prompt: str,
    system: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """Convenience function for LLM completion"""
    manager = await get_llm_manager()
    return await manager.complete(prompt, system, **kwargs)


async def stream(
    prompt: str,
    system: Optional[str] = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Convenience function for LLM streaming"""
    manager = await get_llm_manager()
    async for chunk in manager.stream(prompt, system, **kwargs):
        yield chunk
