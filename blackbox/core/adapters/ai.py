# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
BBX AI Adapter (Phase 5)

Provides AI/LLM integration for:
- Code generation from natural language
- Template enhancement
- Documentation generation
- Code review and suggestions

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (Ollama, LM Studio)

Examples:
    # Generate code from description
    - id: generate_component
      mcp: bbx.ai
      method: generate_code
      inputs:
        prompt: "Create a React button component with hover effects"
        language: "typescript"
        
    # Review code
    - id: review_code
      mcp: bbx.ai
      method: review_code
      inputs:
        code: "${steps.read_file.result}"
        language: "python"
"""

import os
import json
from typing import Dict, Any, List
import httpx


class AIAdapter:
    """BBX Adapter for AI/LLM integration"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.getenv("BBX_AI_MODEL", "gpt-4")
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute AI method"""
        
        if method == "generate_code":
            return await self._generate_code(inputs)
        elif method == "review_code":
            return await self._review_code(inputs)
        elif method == "generate_docs":
            return await self._generate_docs(inputs)
        elif method == "explain_code":
            return await self._explain_code(inputs)
        elif method == "suggest_improvements":
            return await self._suggest_improvements(inputs)
        elif method == "generate_tests":
            return await self._generate_tests(inputs)
        elif method == "chat":
            return await self._chat(inputs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Any:
        """
        Call LLM API
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.api_key:
            return {
                "status": "error",
                "error": "OPENAI_API_KEY not set. Please set it in environment variables."
            }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return {
                        "status": "error",
                        "error": f"API error: {response.status_code} - {response.text}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Request failed: {str(e)}"
            }
    
    async def _generate_code(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code from natural language
        
        Inputs:
            prompt: Code description
            language: Target language
            style: Code style (optional)
            framework: Framework to use (optional)
        """
        prompt = inputs["prompt"]
        language = inputs.get("language", "python")
        framework = inputs.get("framework", "")
        style = inputs.get("style", "clean and professional")
        
        system_prompt = f"""You are an expert {language} developer.
Generate clean, production-ready code based on the user's description.
{"Use " + framework + " framework." if framework else ""}
Code should be {style}.
Return ONLY the code, no explanations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        result = await self._call_llm(messages, temperature=0.3)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "code": result,
            "language": language,
            "prompt": prompt
        }
    
    async def _review_code(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review code and provide suggestions
        
        Inputs:
            code: Code to review
            language: Programming language
            focus: Review focus (security, performance, style)
        """
        code = inputs["code"]
        language = inputs.get("language", "auto-detect")
        focus = inputs.get("focus", "general best practices")
        
        system_prompt = f"""You are an expert code reviewer.
Review the following {language} code with focus on {focus}.
Provide specific, actionable feedback in this JSON format:
{{
    "issues": [
        {{"severity": "high|medium|low", "line": number, "description": "...", "suggestion": "..."}},
        ...
    ],
    "overall_score": 0-100,
    "summary": "brief summary"
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Code to review:\n```{language}\n{code}\n```"}
        ]
        
        result = await self._call_llm(messages, temperature=0.2)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        try:
            review_data = json.loads(result)
            return {
                "status": "success",
                "review": review_data
            }
        except json.JSONDecodeError:
            return {
                "status": "success",
                "review": {
                    "raw_response": result
                }
            }
    
    async def _generate_docs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate documentation for code
        
        Inputs:
            code: Code to document
            language: Programming language
            format: Documentation format (markdown, rst, docstring)
        """
        code = inputs["code"]
        language = inputs.get("language", "python")
        doc_format = inputs.get("format", "markdown")
        
        system_prompt = f"""You are a technical writer.
Generate {doc_format} documentation for the following {language} code.
Include: purpose, parameters, return values, examples.
Be concise but complete."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"```{language}\n{code}\n```"}
        ]
        
        result = await self._call_llm(messages, temperature=0.3)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "documentation": result,
            "format": doc_format
        }
    
    async def _explain_code(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Explain what code does"""
        code = inputs["code"]
        language = inputs.get("language", "auto")
        
        messages = [
            {"role": "system", "content": "Explain code clearly and concisely."},
            {"role": "user", "content": f"Explain this {language} code:\n```\n{code}\n```"}
        ]
        
        result = await self._call_llm(messages)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "explanation": result
        }
    
    async def _suggest_improvements(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest code improvements"""
        code = inputs["code"]
        language = inputs.get("language", "python")
        
        messages = [
            {"role": "system", "content": "Suggest specific code improvements."},
            {"role": "user", "content": f"Suggest improvements for:\n```{language}\n{code}\n```"}
        ]
        
        result = await self._call_llm(messages)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "suggestions": result
        }
    
    async def _generate_tests(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unit tests for code"""
        code = inputs["code"]
        language = inputs.get("language", "python")
        framework = inputs.get("framework", "pytest" if language == "python" else "jest")
        
        system_prompt = f"""Generate comprehensive unit tests for the given code.
Use {framework} framework.
Include edge cases and error scenarios.
Return ONLY the test code."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate tests for:\n```{language}\n{code}\n```"}
        ]
        
        result = await self._call_llm(messages, temperature=0.3)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "tests": result,
            "framework": framework
        }
    
    async def _chat(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """General chat completion"""
        message = inputs["message"]
        system = inputs.get("system", "You are a helpful assistant.")
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]
        
        result = await self._call_llm(messages)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        return {
            "status": "success",
            "response": result
        }
