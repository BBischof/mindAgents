"""Groq integration module."""

import json
from typing import Any
import logging

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

        logging.debug("Groq completion_params: %s", json.dumps(completion_params, indent=2))  

        try:
            # Make API call
            response = self.client.chat.completions.create(**completion_params)
            logging.debug("Groq response: %s", response)  

            # Extract content and tool calls from response
            message = response.choices[0].message
            content = message.content or ""  # Handle None content
            
            # Clean content if it's wrapped in markdown code blocks
            if content.startswith('```') and content.endswith('```'):
                # Remove the markdown code block markers and any language identifier
                content_lines = content.split('\n')
                if len(content_lines) > 2:  # At least 3 lines (opening, content, closing)
                    content = '\n'.join(content_lines[1:-1])  # Remove first and last lines
                    logging.debug("Cleaned markdown content: %s", content)
            
            # Handle tool calls from response
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = [{
                    "tool": tc.function.name,
                    "parameters": json.loads(tc.function.arguments)
                } for tc in message.tool_calls]
            else:
                # Try to parse tool calls from content if not in response
                try:
                    tool_data = json.loads(content)
                    logging.debug("Groq parsed tool_data: %s", tool_data)
                    if isinstance(tool_data, dict) and "name" in tool_data:
                        # Handle the case where the model returns a single tool call as JSON
                        tool_calls = [{
                            "tool": tool_data["name"],
                            "parameters": tool_data.get("arguments", {})
                        }]
                    elif isinstance(tool_data, list):
                        # Handle the case where the model returns multiple tool calls
                        tool_calls = [{
                            "tool": tc.get("name", ""),
                            "parameters": tc.get("arguments", {})
                        } for tc in tool_data if "name" in tc]
                except json.JSONDecodeError as e:
                    logging.debug("Could not parse content as JSON: %s", str(e))
                    pass

            return Response(
                content=content,
                raw_response=response.model_dump(),
                success=True,
                tool_calls=tool_calls
            )

        except Exception as e:
            logging.error("Groq error: %s", str(e))  
            return Response(
                content="",
                raw_response={},
                success=False,
                error=str(e),
            )
