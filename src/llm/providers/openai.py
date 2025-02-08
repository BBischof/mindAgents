"""ChatGPT integration module."""

import json
from typing import Any

from openai import OpenAI

from ..types import LLM, LLMConfig, PromptTemplate, Response, ToolSpec


class ChatGPT(LLM):
    """ChatGPT integration."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize ChatGPT integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key)

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
        """Generate a response from ChatGPT.

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
            # Convert tools to OpenAI format
            openai_tools = []
            for tool in template.available_tools:
                if not isinstance(tool, ToolSpec):
                    continue
                openai_tools.append(
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
            completion_params["tools"] = openai_tools
            completion_params["tool_choice"] = "required"

        try:
            completion = self.client.chat.completions.create(**completion_params)
            message = completion.choices[0].message
            content = message.content or ""

            # Extract tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for call in message.tool_calls:
                    tool_calls.append(
                        {
                            "tool": call.function.name,
                            "parameters": json.loads(call.function.arguments),
                        }
                    )

            return Response(
                content=content,
                raw_response=completion.model_dump(),
                success=True,
                error=None,
                tool_calls=tool_calls,
            )
        except Exception as e:
            return Response(
                content="",
                raw_response={"error": str(e)},
                success=False,
                error=str(e),
                tool_calls=None,
            )
