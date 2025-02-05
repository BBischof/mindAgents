"""Gemini integration module."""

import json
from typing import Any

import google.generativeai as genai

from .base import LLM, LLMConfig, Response
from .types import PromptTemplate, ToolSpec


class Gemini(LLM):
    """Gemini integration."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Gemini integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        genai.configure(api_key=config.api_key)

    async def validate_api_key(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            bool: True if API key is valid, False otherwise.
        """
        try:
            model = genai.GenerativeModel(self.config.model_endpoint)
            response = model.generate_content("test")
            return bool(response)
        except Exception:
            return False

    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """Generate a response from Gemini.

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

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            gemini_messages.append(
                {
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}],
                }
            )

        data = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": template.temperature,
                "maxOutputTokens": self.config.max_tokens if self.config.max_tokens else 2048,
            },
        }

        # Add tools configuration if template has tools
        if template.available_tools:
            # Convert tools to Gemini format
            gemini_tools = []
            for tool in template.available_tools:
                if not isinstance(tool, ToolSpec):
                    continue
                gemini_tools.append(
                    {
                        "functionDeclarations": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": tool.parameters,
                                    "required": tool.required_params,
                                },
                            }
                        ]
                    }
                )
            data["tools"] = gemini_tools
            data["toolChoice"] = "required"  # Force tool usage

        try:
            model = genai.GenerativeModel(self.config.model_endpoint)
            response = model.generate_content(data)

            # Extract tool calls if present
            tool_calls = None
            if hasattr(response, "candidates") and response.candidates[0].content.parts[0].function_call:
                tool_calls = []
                function_call = response.candidates[0].content.parts[0].function_call
                tool_calls.append(
                    {
                        "tool": function_call.name,
                        "parameters": json.loads(function_call.args),
                    }
                )

            return Response(
                content=str(response.text) if response.text else "",  # Handle None content
                raw_response=response._raw_response,
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
