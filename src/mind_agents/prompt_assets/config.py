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
        ValueError: If config values are not strings
    """
    config_path = Path.home() / ".config" / "llm_keys" / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. " "Please create it with your API keys.")

    with open(config_path) as f:
        raw_config = json.load(f)

    # Validate and convert config to dict[str, str]
    config: dict[str, str] = {}
    for key, value in raw_config.items():
        if not isinstance(key, str):
            raise ValueError(f"Config key {key} must be a string")
        if not isinstance(value, str):
            raise ValueError(f"Config value for {key} must be a string")
        config[key] = value

    required_keys = {
        "openai_api_key": "OpenAI",
        "anthropic_api_key": "Anthropic",
        "google_api_key": "Google",
    }

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
