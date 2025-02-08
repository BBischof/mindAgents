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
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            return kwargs
    except Exception as e:
        logger.error(f"AST parsing failed for tool arguments: {e}")
    return {}


class Gemini(LLM):
    """Gemini integration using the new Model configuration."""

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize Gemini integration.
        Args:
            config: LLM configuration.
        """
        super().__init__(config)
        # Use the API key from the config and any transport settings.
        genai.configure(api_key=config.api_key, transport="rest")
        # Note: the model is not created here since the generation configuration (e.g. temperature)
        # may depend on the prompt template.

    async def validate_api_key(self) -> bool:
        """
        Validate the API key by making a test request.
        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            model = genai.GenerativeModel(
                model_name=self.config.metadata.model_id,
                generation_config={"temperature": 0.1},
            )
            response = model.generate_content("test")
            return bool(response)
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    async def generate(
        self,
        template: PromptTemplate,
        dynamic_content: dict[str, Any],
    ) -> Response:
        """
        Generate a response from Gemini using the new model configuration.
        The prompt is built from the provided template and dynamic content.
        Depending on the model:
          - For Gemini 1.5 models: a separate tool_config argument is passed to force a function call
            (using ANY mode with allowed_function_names derived from the available tools).
          - For Gemini 2.0 models: explicit prompt instructions are appended to force a tool call.
        Args:
            template: The prompt template containing components, tools, and configuration.
            dynamic_content: Values to fill placeholders in the template.
        Returns:
            Response: Generated response with optional tool call information.
        """
        try:
            generation_params: dict[str, Any] = {"temperature": template.temperature}
            if self.config.max_tokens is not None:
                generation_params["max_tokens"] = self.config.max_tokens
            if (
                getattr(self.config.metadata, "supports_reasoning_effort", False)
                and self.config.metadata.reasoning_effort is not None
            ):
                generation_params["reasoning_effort"] = self.config.metadata.reasoning_effort.value

            # For Gemini 1.5 models, prepare a separate tool_config
            forced_tool_config: dict[str, Any] | None = None
            if self.config.model_name.startswith("GEMINI_1_5") and template.available_tools:
                forced_tool_config = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [
                            tool.name for tool in template.available_tools if isinstance(tool, ToolSpec)
                        ],
                    }
                }

            # Create the Gemini model using the new model identifier.
            model = genai.GenerativeModel(
                model_name=self.config.metadata.model_id,
                generation_config=generation_params,
            )

            # Construct messages from the prompt template.
            messages = template.construct_prompt(dynamic_content)
            if not getattr(self.config.metadata, "supports_system_messages", True):
                system_content = ""
                user_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system_content += msg.get("content", "") + "\n\n"
                    else:
                        user_messages.append(msg)
                if system_content and user_messages:
                    user_messages[0]["content"] = system_content + user_messages[0]["content"]
                messages = user_messages

            prompt = "\n".join(msg.get("content", "") for msg in messages)

            # Append tool information to the prompt.
            if template.available_tools:
                prompt += "\n\nAvailable tools:\n"
                for tool in template.available_tools:
                    if isinstance(tool, ToolSpec):
                        prompt += f"\n{tool.name}: {tool.description}\n"
                        prompt += "Parameters:\n"
                        for param_name, param_info in tool.parameters.items():
                            prompt += f"- {param_name}: {param_info.get('description', '')}\n"

            # For Gemini 2.0 models, add explicit instructions.
            if self.config.model_name == "GEMINI_2_0_FLASH" and template.available_tools:
                prompt += (
                    "\n\nYou are required to respond with a function call. Your response "
                    "must be in JSON format with exactly two keys: 'tool' and 'parameters'. "
                    "Do not include any additional text."
                )

            logger.debug(f"Sending prompt to Gemini: {prompt}")
            logger.debug(f"Generation config: {generation_params}")

            # Call the Gemini API.
            # For Gemini 1.5 models, pass the forced tool config as a separate parameter.
            if forced_tool_config is not None:
                response = model.generate_content(prompt, tool_config=forced_tool_config)
            else:
                response = model.generate_content(prompt)

            logger.debug(f"Got raw response from Gemini: {response}")

            content = response.text if hasattr(response, "text") else str(response)
            logger.debug(f"Extracted content: {content}")

            tool_calls = None
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

            if not tool_calls:
                match = re.search(r"```(?:TOOL_CALL|tool_code)\s+(.*?)```", content, re.DOTALL)
                if match:
                    tool_call_str = match.group(1).strip()
                    logger.debug(f"Found TOOL_CALL block: {tool_call_str}")
                    func_match = re.match(r"(\w+)\((.*)\)", tool_call_str)
                    if func_match:
                        tool_name = func_match.group(1)
                        args_str = func_match.group(2)
                        tool_data = {"tool": tool_name}
                        parsed_args = parse_tool_arguments(args_str)
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
                raw_response=getattr(response, "_raw_response", {}),
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
