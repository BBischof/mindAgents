"""Groq integration module."""

import json
from typing import Any

from groq import Groq

from ..types import LLM, LLMConfig, PromptTemplate, Response, ToolSpec


class GroqChat(LLM):
    """Groq integration."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Groq integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        self.client = Groq(api_key=config.api_key)

    async def validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """Generate a response from Groq.

        Args:
            template: The prompt template containing components, tools, and configuration
            dynamic_content: Values to fill placeholders in components

        Returns:
            Response: Generated response.

        Raises:
            Exception: If API request fails.
        """
        # Use template to construct messages
        messages = template.construct_prompt(dynamic_content)

        # Prepare completion parameters
        completion_params: dict[str, Any] = {
            "model": self.config.metadata.model_id,
            "messages": messages,
        }

        # Add temperature if supported
        if self.config.metadata.supports_temperature:
            completion_params["temperature"] = template.temperature

        if self.config.max_tokens is not None:
            completion_params["max_tokens"] = self.config.max_tokens

        try:
            # Make API call
            response = self.client.chat.completions.create(**completion_params)

            # Extract content from response
            content = response.choices[0].message.content

            return Response(
                content=content,
                raw_response=response.model_dump(),
                success=True,
            )

        except Exception as e:
            return Response(
                content="",
                raw_response={},
                success=False,
                error=str(e),
            )
