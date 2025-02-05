"""Prompt templates for various LLM interactions."""


from mind_agents.prompt_assets.utilities import generate_play_content

from .wait_n_seconds_prompts import play_game_template

__all__ = ["play_game_template", "generate_play_content"]
