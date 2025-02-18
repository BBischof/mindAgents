"""Claude integration module."""

import logging
from typing import Any

import anthropic

from ..types import LLM, LLMConfig, PromptTemplate, Response, ToolSpec

logger = logging.getLogger(__name__)


class Claude(LLM):
    """Claude integration."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Claude integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.api_key)

    async def validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            # List models to validate the API key
            self.client.models.list()
            return True
        except Exception:
            return False

    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """Generate a response from Claude.

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

        # Convert messages to Claude format and find system message
        claude_messages = []
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append(
                    {"role": "assistant" if msg["role"] == "assistant" else "user", "content": msg["content"]}
                )

        # Prepare tools if available
        tools = None
        if template.available_tools:
            tools = []
            for tool in template.available_tools:
                if not isinstance(tool, ToolSpec):
                    continue
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required_params,
                        },
                    }
                )

        try:
            # Make the API call using the client
            message_params = {
                "model": self.config.metadata.model_id,
                "messages": claude_messages,
                "system": system_message,
                "temperature": template.temperature,
                "tools": tools,
                "max_tokens": self.config.max_tokens
                if isinstance(self.config.max_tokens, int) and self.config.max_tokens > 0
                else 1024,
            }

            # Log what we're about to send
            logger.debug(
                f"Sending to Claude API:\nSystem: {system_message}\nMessages: {claude_messages}\nTools: {tools}"
            )

            # Make the API call using the client
            message = self.client.messages.create(**message_params)

            # Log the response for debugging
            logger.debug(f"Response from Claude API: {message}")

            # Extract content and tool calls
            content = ""
            tool_calls = None

            # Process each content block
            for block in message.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls = [{"tool": block.name, "parameters": block.input}]

            return Response(
                content=content,
                raw_response=message.model_dump(),
                success=True,
                error=None,
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.error(f"Error from Claude API: {str(e)}")
            return Response(
                content="",
                raw_response={"error": str(e)},
                success=False,
                error=str(e),
                tool_calls=None,
            )
