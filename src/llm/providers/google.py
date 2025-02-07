"""Gemini integration module."""

import ast
import json
import logging
import re
from typing import Any

import google.generativeai as genai

from ..types import LLM, LLMConfig, PromptTemplate, Response, ToolSpec

logger = logging.getLogger(__name__)


def parse_tool_arguments(arg_str: str) -> dict:
    """
    Parses a function-call style argument string (e.g., "seconds=5, reason='test'")
    and returns a dictionary of keyword arguments.

    This function wraps the argument string in a fake function call so that it can be
    parsed by Python's AST module.
    """
    fake_call = "f(" + arg_str + ")"
    try:
        node = ast.parse(fake_call, mode="eval")
        if isinstance(node, ast.Expression) and isinstance(node.body, ast.Call):
            call_node = node.body
            kwargs = {}
            for keyword in call_node.keywords:
                # Safely evaluate each keyword argument's value.
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            return kwargs
    except Exception as e:
        logger.error(f"AST parsing failed for tool arguments: {e}")
    return {}


class Gemini(LLM):
    """Gemini integration."""

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize Gemini integration.

        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        genai.configure(api_key=config.api_key, transport="rest")
        # Don't create the model here since we need the template's temperature
        # for configuration

    async def validate_api_key(self) -> bool:
        """
        Validate the API key by making a test request.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            # Create a temporary model just for validation
            model = genai.GenerativeModel(
                model_name=self.config.model_endpoint,
                generation_config={"temperature": 0.1},  # Use default temp for validation
            )
            response = model.generate_content("test")
            return bool(response)
        except Exception:
            return False

    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """
        Generate a response from Gemini.

        Constructs the prompt from the provided template and dynamic content, appends tool
        descriptions if available, and then sends the prompt to Gemini. It then attempts to
        extract any tool call information from the Gemini response.

        Args:
            template: The prompt template containing components, tools, and configuration.
            dynamic_content: Values to fill placeholders in the template.

        Returns:
            Response: Generated response with optional tool call information.
        """
        try:
            # Create model with template's temperature
            model = genai.GenerativeModel(
                model_name=self.config.model_endpoint,
                generation_config={"temperature": template.temperature},
            )

            # Build the prompt by concatenating the message contents.
            prompt = ""
            messages = template.construct_prompt(dynamic_content)
            for msg in messages:
                if msg.get("role") == "system":
                    prompt = f"{msg.get('content')}\n\n{prompt}"
                else:
                    prompt += f"{msg.get('content')}\n"

            logger.debug(f"Sending prompt to Gemini: {prompt}")

            # Append available tool information to the prompt.
            if template.available_tools:
                prompt += "\n\nAvailable tools:\n"
                for tool in template.available_tools:
                    if isinstance(tool, ToolSpec):
                        prompt += f"\n{tool.name}: {tool.description}\n"
                        prompt += "Parameters:\n"
                        for param_name, param_info in tool.parameters.items():
                            prompt += f"- {param_name}: {param_info.get('description', '')}\n"

            # Call the Gemini API
            response = model.generate_content(prompt)
            logger.debug(f"Got raw response from Gemini: {response}")

            # Extract the text content.
            content = response.text if hasattr(response, "text") else str(response)
            logger.debug(f"Extracted content: {content}")

            tool_calls = None

            # Attempt to extract a JSON-like tool call structure from the response.
            try:
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    tool_data = json.loads(json_str)
                    if "tool" in tool_data and "parameters" in tool_data:
                        tool_calls = [tool_data]
                        logger.debug(f"Extracted tool call using JSON: {tool_calls}")
            except Exception as e:
                logger.debug("No valid JSON tool calls found in response: " + str(e))

            # Fallback: try to extract a tool call from a markdown code block.
            if not tool_calls:
                match = re.search(r"```TOOL_CALL\s+(.*?)```", content, re.DOTALL)
                if match:
                    tool_call_str = match.group(1).strip()
                    logger.debug(f"Found TOOL_CALL block: {tool_call_str}")
                    func_match = re.match(r"(\w+)\((.*)\)", tool_call_str)
                    if func_match:
                        tool_name = func_match.group(1)
                        args_str = func_match.group(2)
                        tool_data = {"tool": tool_name}
                        parsed_args = parse_tool_arguments(args_str)
                        # If a tool spec is provided in the template, check for missing required parameters.
                        if template.available_tools is not None:
                            for t in template.available_tools:
                                if t.name == tool_name:
                                    missing = [p for p in t.required_params if p not in parsed_args]
                                    if missing:
                                        logger.error(f"Missing required parameter(s) {missing} for tool {tool_name}")
                                    break
                        tool_data["parameters"] = parsed_args
                        tool_calls = [tool_data]
                        logger.debug(f"Extracted tool call using regex: {tool_calls}")

            return Response(
                content=content,
                raw_response=response._raw_response if hasattr(response, "_raw_response") else {},
                success=True,
                error=None,
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Error in Gemini generate: {e}")
            return Response(
                content="",
                raw_response={"error": str(e)},
                success=False,
                error=str(e),
                tool_calls=None,
            )
