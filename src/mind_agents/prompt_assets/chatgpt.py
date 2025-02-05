"""ChatGPT integration module."""

import json
from typing import Any

import httpx

from .base import LLM, LLMConfig, Response
from .types import PromptTemplate, ToolSpec


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

        data = {
            "model": self.config.model_endpoint,
            "messages": messages,
            "temperature": template.temperature,  # Use template's temperature
        }
        if self.config.max_tokens is not None:
            data["max_tokens"] = self.config.max_tokens

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
            data["tools"] = openai_tools
            data["tool_choice"] = "required"  # Force tool usage

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    json=data,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Extract tool calls if present
                tool_calls = None
                if "tool_calls" in result["choices"][0]["message"]:
                    tool_calls = []
                    for call in result["choices"][0]["message"]["tool_calls"]:
                        tool_calls.append(
                            {
                                "tool": call["function"]["name"],
                                "parameters": json.loads(call["function"]["arguments"]),
                            }
                        )

                return Response(
                    content=content or "",  # Handle None content
                    raw_response=result,
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
