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

        # For models that don't support system messages, convert them to user messages
        if not self.config.metadata.supports_system_messages:
            # Combine any system messages into the first user message
            system_content = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content += msg["content"] + "\n\n"
                else:
                    user_messages.append(msg)

            if system_content and user_messages:
                user_messages[0]["content"] = system_content + user_messages[0]["content"]
            messages = user_messages

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

        # Add reasoning_effort if supported and specified
        if self.config.metadata.supports_reasoning_effort and self.config.metadata.reasoning_effort is not None:
            completion_params["reasoning_effort"] = self.config.metadata.reasoning_effort.value

        # Add tools configuration if template has tools
        if template.available_tools:
            # Convert tools to function format
            tools = []
            for tool in template.available_tools:
                if not isinstance(tool, ToolSpec):
                    continue
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.parameters,
                                "required": tool.required_params,
                            },
                        },
                    }
                )
            if tools:
                completion_params["tools"] = tools

        try:
            # Make API call
            response = self.client.chat.completions.create(**completion_params)

            # Extract content from response
            content = response.choices[0].message.content

            # Try to parse tool calls from JSON content
            tool_calls = None
            try:
                tool_data = json.loads(content)
                if isinstance(tool_data, dict):
                    tool_calls = [{
                        "tool": tool_data.get("name", ""),
                        "parameters": tool_data.get("arguments", {})
                    }]
            except json.JSONDecodeError:
                pass

            return Response(
                content=content,
                raw_response=response.model_dump(),
                success=True,
                tool_calls=tool_calls
            )

        except Exception as e:
            return Response(
                content="",
                raw_response={},
                success=False,
                error=str(e),
            )
