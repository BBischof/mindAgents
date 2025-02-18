"""Functions for generating prompt content and managing LLM clients."""

import json
import os
from pathlib import Path
from typing import Any, Optional, cast
import logging

import tiktoken

from .providers.anthropic import Claude
from .providers.google import Gemini
from .providers.groq import GroqChat
from .providers.openai import ChatGPT
from .types import LLM, LLMConfig, get_model_metadata

logger = logging.getLogger(__name__)

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
        return len(encoding.encode(str(text["text"])))
    return 0


def get_model_implementation(model_name: str) -> type[LLM]:
    """Get the appropriate LLM implementation for a model.

    Args:
        model_name: The name of the model to get the implementation for

    Returns:
        The LLM implementation class
    """
    provider = get_model_metadata(model_name).provider
    if provider == "openai":
        return ChatGPT
    elif provider == "anthropic":
        return Claude
    elif provider == "google":
        return Gemini
    elif provider == "groq":
        return GroqChat
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_api_key(model_name: str) -> str:
    """Load API key from environment variables or config file.

    Args:
        model_name: The name of the model to get the API key for

    Returns:
        The API key for the model

    Raises:
        ValueError: If API key not found in environment variables or config file
    """
    provider = get_model_metadata(model_name).provider
    env_key = f"{provider.upper()}_API_KEY"
    
    # First try environment variable
    api_key = os.getenv(env_key)
    if api_key:
        return api_key
        
    # Fall back to config file
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"
    if not config_path.exists():
        raise ValueError(
            f"Neither environment variable {env_key} is set nor config file exists at {config_path}. "
            "Please set the environment variable or create the config file with your API keys."
        )

    with open(config_path) as f:
        config = json.load(f)

    key_name = f"{provider}_api_key"
    if key_name not in config:
        raise ValueError(
            f"API key not found. Please either:\n"
            f"1. Set the {env_key} environment variable, or\n"
            f"2. Add '{key_name}' to your config file at {config_path}"
        )
    return cast(str, config[key_name])


async def get_llm_client(model_name: str, api_key: Optional[str] = None) -> LLM:
    """Get an LLM client for the specified model.

    Args:
        model_name: The name of the model to use
        api_key: Optional API key. If not provided, will be loaded from config

    Returns:
        An initialized LLM client
    """
    implementation = get_model_implementation(model_name)
    if api_key is None:
        api_key = load_api_key(model_name)

    # Get model metadata
    metadata = get_model_metadata(model_name)
    provider = metadata.provider

    # Load config to check for max_tokens override
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"
    max_tokens = None

    # Try environment variable first
    env_var = f"{provider.upper()}_MAX_TOKENS"
    env_value = os.getenv(env_var)
    if env_value:
        try:
            max_tokens = int(env_value)
        except ValueError:
            logger.warning(f"Invalid {env_var} value: {env_value}. Must be an integer.")

    # If no env var, try config file
    if max_tokens is None and config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                max_tokens_key = f"{provider}_max_tokens"
                if max_tokens_key in config:
                    try:
                        max_tokens = int(config[max_tokens_key])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid {max_tokens_key} in config: {config[max_tokens_key]}. Must be an integer.")
        except (json.JSONDecodeError, IOError):
            pass

    # If no config found, use model's default
    if max_tokens is None:
        max_tokens = metadata.default_max_tokens

    config = LLMConfig(
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens
    )
    return implementation(config)


def generate_play_content(
    card_number: int,
    num_players: int,
    total_other_cards: int,
    all_cards: list[int],
    played_cards: list[int] | None = None,
) -> dict[str, Any]:
    """Generate content for playing a card.

    Args:
        card_number: The number on the card (1-100)
        num_players: Number of players in the game
        total_other_cards: Total number of cards held by other players
        all_cards: List of all cards in the player's hand
        played_cards: List of cards already played in ascending order

    Returns:
        Dictionary containing the dynamic content for the prompt
    """
    if played_cards is None:
        played_cards = []

    return {
        "card_number": card_number,
        "num_players": num_players,
        "total_other_cards": total_other_cards,
        "all_cards": all_cards,
        "played_cards": played_cards,
    }
