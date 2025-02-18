"""Configuration utilities for LLM integrations."""

import json
from pathlib import Path
from typing import Optional


def load_api_keys() -> dict[str, str]:
    """Load API keys from config file.

    Returns:
        Dict containing API keys for different providers

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required keys are missing from config
        ValueError: If API key values are not strings
    """
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. " "Please create it with your API keys.")

    with open(config_path) as f:
        raw_config = json.load(f)

    # Only validate API key values
    required_keys = {
        "openai_api_key": "OpenAI",
        "anthropic_api_key": "Anthropic",
        "google_api_key": "Google",
        "groq_api_key": "Groq",
    }

    config: dict[str, str] = {}
    for key in required_keys:
        if key not in raw_config:
            continue
        value = raw_config[key]
        if not isinstance(value, str):
            raise ValueError(f"API key value for {key} must be a string")
        config[key] = value

    missing_keys = [key for key in required_keys if key not in config or not config[key]]

    if missing_keys:
        missing_providers = [required_keys[key] for key in missing_keys]
        raise KeyError(f"Missing API keys for: {', '.join(missing_providers)}. " f"Please add them to {config_path}")

    return config


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider.

    Args:
        provider: The provider to get the key for (openai, anthropic, or google)

    Returns:
        The API key if found, None otherwise
    """
    try:
        config = load_api_keys()
        key_name = f"{provider}_api_key"
        return config.get(key_name)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {str(e)}")
        return None
