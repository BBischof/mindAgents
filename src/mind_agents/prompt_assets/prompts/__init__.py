"""Prompt templates for various LLM interactions."""

from typing import Any

from mind_agents.prompt_assets.types import Model

from .play_game import generate_play_content, play_game_template

__all__ = ["play_game_template", "generate_play_content"]


def get_prompt_for_model(model: Model, state: dict[str, Any]) -> str:
    """Get the appropriate prompt for the given model and game state.

    Args:
        model: The model to generate the prompt for
        state: The current game state containing:
            "1": List of player's cards
            "H": Total number of cards held by other players
            "P": List of played cards

    Returns:
        A string containing the prompt for the model
    """
    # For now, all models use the same prompt template and content generation
    # In the future, we can add model-specific templates if needed
    content = generate_play_content(
        min(state["1"]),  # Use lowest card as the one to play
        1,  # Only one other player in test scenarios
        state["H"],  # Total cards held by others
        played_cards=state.get("P", []),  # Cards already played
    )

    messages = play_game_template.construct_prompt(content)
    # For now, just return the content of the last message
    # In the future, we might want to handle this differently
    last_message = messages[-1]
    if not isinstance(last_message, dict) or "content" not in last_message:
        raise ValueError("Expected last message to be a dict with 'content' key")
    content = last_message["content"]
    if not isinstance(content, str):
        raise ValueError("Expected content to be a string")
    return content
