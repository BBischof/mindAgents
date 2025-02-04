"""ChatGPT integration module."""

from typing import Optional

import httpx

from .base import LLM, LLMConfig, Response


class ChatGPT(LLM):
    """ChatGPT integration."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: LLMConfig) -> None:
        """Initialize ChatGPT integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }

    async def validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers=self.headers,
                    timeout=10.0,
                )
                return bool(response.status_code == 200)
        except Exception:
            return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Response:
        """Generate a response from ChatGPT.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt to set context.

        Returns:
            Response: Generated response.

        Raises:
            Exception: If API request fails.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model_endpoint,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            data["max_tokens"] = self.config.max_tokens

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    json=data,
                    timeout=30.0,
                )
                await response.aread()  # Ensure response is fully read
                await response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                return Response(content=content, raw_response=result)
        except Exception as e:
            if isinstance(e, Exception) and "object Response can't be used in 'await' expression" in str(e):
                # Re-raise without wrapping to avoid double-wrapping
                raise e from None
            raise Exception(f"Failed to generate response: {str(e)}") from e
