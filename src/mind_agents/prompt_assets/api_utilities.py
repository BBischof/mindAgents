"""LLM API utilities."""

import logging
from typing import Any, Optional

import tiktoken

from .base import LLM, Response
from .chatgpt import ChatGPT
from .claude import Claude
from .config import get_api_key
from .gemini import Gemini
from .types import Model, PromptTemplate

# Map of Model enum to their API endpoint strings
MODEL_ENDPOINTS: dict[Model, str] = {
    # OpenAI models
    Model.GPT4: "gpt-4",
    Model.GPT4_TURBO: "gpt-4-turbo-preview",
    Model.GPT35: "gpt-3.5-turbo",
    # Anthropic models
    Model.CLAUDE3_OPUS: "claude-3-opus-20240229",
    Model.CLAUDE3_SONNET: "claude-3-sonnet-20240229",
    # Google models
    Model.GEMINI_PRO: "gemini-pro",
    Model.GEMINI_PRO_VISION: "gemini-pro-vision",
}


def get_token_count(text: str | list | dict, model: str = "gpt-4") -> int:
    """Count tokens in text using the appropriate encoding for the model.

    Args:
        text: The text to count tokens for. Can be a string, list, or dict
        model: The model to use for token counting

    Returns:
        int: Number of tokens in the text
    """
    encoding = tiktoken.encoding_for_model(model)

    if isinstance(text, str):
        return len(encoding.encode(text))
    elif isinstance(text, list):
        return sum(
            len(encoding.encode(str(item["text"])))
            for item in text
            if isinstance(item, dict) and "text" in item and item["text"] is not None
        )
    elif isinstance(text, dict) and "text" in text:
        return len(encoding.encode(str(text["text"]))) if text["text"] is not None else 0
    else:
        logging.warning(f"Unexpected content type: {type(text)}")
        return 0


def get_model_implementation(model: Model) -> type[LLM]:
    """Get the appropriate LLM implementation class for a model.

    Args:
        model: The model to get the implementation for

    Returns:
        Type: The LLM implementation class

    Raises:
        ValueError: If no implementation found for the model
    """
    if model in [Model.GPT4, Model.GPT4_TURBO, Model.GPT35]:
        return ChatGPT
    elif model in [Model.CLAUDE3_OPUS, Model.CLAUDE3_SONNET]:
        return Claude
    elif model in [Model.GEMINI_PRO, Model.GEMINI_PRO_VISION]:
        return Gemini
    else:
        raise ValueError(f"No implementation found for model {model}")


async def get_llm_client(model: Model, api_key: Optional[str] = None) -> LLM:
    """Get an LLM client for the specified model.

    Args:
        model: The model to use
        api_key: Optional API key. If not provided, will try to get from config

    Returns:
        The LLM client instance

    Raises:
        ValueError: If model implementation not found or API key not available
    """
    if model not in MODEL_ENDPOINTS:
        raise ValueError(f"No endpoint found for model {model}")

    # Get API key from config if not provided
    if api_key is None:
        if model in [Model.GPT4, Model.GPT4_TURBO, Model.GPT35]:
            api_key = get_api_key("openai")
        elif model in [Model.CLAUDE3_OPUS, Model.CLAUDE3_SONNET]:
            api_key = get_api_key("anthropic")
        elif model in [Model.GEMINI_PRO, Model.GEMINI_PRO_VISION]:
            api_key = get_api_key("google")

    if not api_key:
        raise ValueError(f"No API key found for model {model}")

    config = type(
        "Config",
        (),
        {
            "api_key": api_key,
            "temperature": 0.7,  # Default values, will be overridden by template
            "max_tokens": None,
            "model_endpoint": MODEL_ENDPOINTS[model],  # Pass the actual endpoint string
        },
    )()

    implementation_class = get_model_implementation(model)
    return implementation_class(config)


async def generate_response(
    messages: list[dict[str, Any]],
    template: PromptTemplate,
) -> Response:
    """Generate a response using the specified template and model.

    Args:
        messages: List of message dictionaries for the chat
        template: The prompt template containing model configuration

    Returns:
        Response: The model's response or error information
    """
    try:
        llm = await get_llm_client(template.model)

        # Convert messages to prompt format
        prompt = messages[-1]["content"]  # User message is always last
        system_prompt = None
        if len(messages) > 1 and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]

        response = await llm.generate(prompt=prompt, system_prompt=system_prompt)
        if not response.content:
            logging.error("Model returned empty response")
        return response

    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        logging.error(error_msg)
        return Response(
            content="",
            raw_response={"error": str(e)},
        )
