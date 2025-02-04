"""Gemini integration module."""

from typing import Any, Optional

import httpx

from .base import LLM, LLMConfig, Response


class Gemini(LLM):
    """Gemini integration."""

    API_URL_BASE = "https://generativelanguage.googleapis.com/v1/models"

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Gemini integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        self.api_url = f"{self.API_URL_BASE}/{config.model_endpoint}:generateContent"
        self.headers = {
            "Content-Type": "application/json",
        }

    async def validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_URL_BASE}?key={self.config.api_key}",
                    headers=self.headers,
                    timeout=10.0,
                )
                return bool(response.status_code == 200)
        except Exception:
            return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Response:
        """Generate a response from Gemini.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt to set context.

        Returns:
            Response: Generated response.

        Raises:
            Exception: If API request fails.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.config.temperature},
        }
        if self.config.max_tokens is not None:
            data["generationConfig"]["maxOutputTokens"] = self.config.max_tokens

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}?key={self.config.api_key}",
                    headers=self.headers,
                    json=data,
                    timeout=30.0,
                )
                await response.aread()  # Ensure response is fully read
                await response.raise_for_status()
                result = await response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return Response(content=content, raw_response=result)
        except Exception as e:
            if isinstance(e, Exception) and "object Response can't be used in 'await' expression" in str(e):
                # Re-raise without wrapping to avoid double-wrapping
                raise e from None
            raise Exception(f"Failed to generate response: {str(e)}") from e
