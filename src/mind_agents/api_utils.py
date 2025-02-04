"""Utilities for interacting with LLM APIs."""

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from mind_agents.prompt_assets.prompts.play_game import AVAILABLE_TOOLS
from mind_agents.prompt_assets.types import (
    Model,
    PromptTemplate,
    Response,
    ToolCall,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Base class for LLM API clients."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the LLM client.

        Args:
            config: Configuration dictionary for the client
        """
        self.config = config

    async def generate_response(self, template: PromptTemplate, content: dict[str, Any]) -> Response:
        """Generate a response using the LLM.

        Args:
            template: The prompt template to use
            content: Dynamic content for the template

        Returns:
            Response object containing the LLM's response
        """
        raise NotImplementedError("Subclasses must implement generate_response")


class ChatGPTClient(LLMClient):
    """Client for OpenAI's ChatGPT API."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the ChatGPT client.

        Args:
            config: Configuration dictionary for the client
        """
        super().__init__(config)
        if "openai_api_key" not in config:
            raise ValueError("openai_api_key not found in config")
        self.client = AsyncOpenAI(api_key=config["openai_api_key"])

    async def generate_response(self, template: PromptTemplate, content: dict[str, Any]) -> Response:
        """Generate a response using ChatGPT.

        Args:
            template: The prompt template to use
            content: Dynamic content for the template

        Returns:
            Response object containing ChatGPT's response
        """
        try:
            messages = template.construct_prompt(content)
            tools: list[dict[str, Any]] = []

            # Add tool definitions if template has tools
            if template.available_tools:
                for tool_name in template.available_tools:
                    if tool_name not in AVAILABLE_TOOLS:
                        continue

                    tool_spec = AVAILABLE_TOOLS[tool_name]
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": tool_spec.name,
                            "description": tool_spec.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool_spec.parameters,
                                "required": tool_spec.required_params,
                            },
                        },
                    }
                    tools.append(tool_def)

            # Make first API call to get reasoning
            first_response: ChatCompletion = await self.client.chat.completions.create(
                model=template.model.value,
                messages=[
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                    for msg in messages
                ],
                temperature=template.temperature,
                top_p=template.top_p,
            )

            # Extract reasoning
            reasoning = first_response.choices[0].message.content

            # Add reasoning to messages
            messages.append(
                {
                    "role": "assistant",
                    "content": reasoning,
                }
            )

            # Add instruction to use tools
            messages.append(
                {
                    "role": "user",
                    "content": "Now use one of the available tools to take action based on your reasoning.",
                }
            )

            # Make second API call to force tool usage
            second_response: ChatCompletion = await self.client.chat.completions.create(
                model=template.model.value,
                messages=messages,
                temperature=template.temperature,
                top_p=template.top_p,
                tools=tools if tools else None,
                tool_choice="required" if tools else None,  # Force tool usage
            )

            # Extract response content and tool calls
            message = second_response.choices[0].message
            tool_calls = None

            if message.tool_calls:
                tool_calls = []
                for call in message.tool_calls:
                    try:
                        tool_calls.append(
                            ToolCall(
                                tool=call.function.name,
                                parameters=json.loads(call.function.arguments),
                            )
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool call arguments: {e}")
                        continue

            return Response(
                content=reasoning,  # Use the reasoning from first call
                raw_response=second_response.model_dump(),
                success=True,
                error=None,
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return Response(
                content=None,
                raw_response=None,
                success=False,
                error=str(e),
                tool_calls=None,
            )


def get_model_implementation(model: Model) -> type[LLMClient]:
    """Get the appropriate LLM client implementation for a model.

    Args:
        model: The model to get an implementation for

    Returns:
        The LLM client class for the model

    Raises:
        ValueError: If no implementation exists for the model
    """
    implementations = {
        Model.GPT4: ChatGPTClient,
        Model.GPT35: ChatGPTClient,
    }

    if model not in implementations:
        raise ValueError(f"No implementation available for model {model}")

    return implementations[model]


def get_llm_client(config: Optional[dict[str, Any]] = None) -> LLMClient:
    """Get a configured LLM client.

    Args:
        config: Optional configuration for the client

    Returns:
        A configured LLM client
    """
    if config is None:
        config = {}  # Use default config

    # For now, always use ChatGPT
    client_class = get_model_implementation(Model.GPT4)
    return client_class(config)
