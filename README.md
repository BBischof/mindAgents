# The Mind - AI Card Game

A Python implementation of The Mind card game where AI agents collaborate to play cards in ascending order without communication.

## Game Overview

The Mind is a cooperative card game where players must play their cards in ascending order without any verbal communication. In this implementation, AI agents take on the role of players, using sophisticated language models to make decisions about when to play their cards.

### Core Features
- Multiple AI players working together
- Lives system for mistakes
- Bonus lives awarded at specific rounds
- Star power for revealing cards
- Auto-play system for maintaining game flow

## Project Structure

The project is organized into the following main components:

```
src/
├── core/                    # Core game logic
│   ├── game.py             # Game mechanics and state
│   ├── display.py          # Display utilities
│   └── simulator.py        # Game simulation tools
│
├── llm/                    # LLM integration
│   ├── providers/          # LLM providers
│   │   ├── anthropic.py    # Claude integration
│   │   ├── openai.py       # GPT integration
│   │   └── google.py       # Gemini integration
│   ├── prompts/           # LLM prompts
│   │   └── wait_n_seconds_prompts.py
│   ├── types.py           # Type definitions
│   └── utilities.py       # LLM utilities
│
└── play.py               # Main entry point
```

## Model Selection

The game supports multiple AI models that can be assigned to different players. Each model may have different strategies and behaviors.

### Available Models
- OpenAI Models:
  - GPT4: GPT-4 (8k context)
  - GPT4_TURBO: GPT-4 Turbo (latest function-calling enabled)
  - GPT35_TURBO: GPT-3.5 Turbo (4k context)
  - GPT_O1: OpenAI o1 reasoning model
  - GPT_O3_MINI: OpenAI o3-mini model

- Anthropic Models:
  - CLAUDE3_OPUS: Claude 3 Opus model
  - CLAUDE3_SONNET: Claude 3 Sonnet model
  - CLAUDE3_HAIKU: Claude 3 Haiku model
  - CLAUDE35_SONNET: Claude 3.5 Sonnet model
  - CLAUDE35_HAIKU: Claude 3.5 Haiku model

- Google Models:
  - GEMINI_1_5_FLASH: Gemini 1.5 Flash model
  - GEMINI_1_5_PRO: Gemini 1.5 Pro model
  - GEMINI_2_0_FLASH: Gemini 2.0 Flash model

### Using Different Models

You can specify which model each player should use when running the game. To see all available models and their descriptions:

```bash
# List all available models
uv run python -m src.play --list-models
# or for the simulator
uv run python -m src.core.simulator --list-models
```

Then use specific models in your games:

```bash
# Run game with specific models for each player
uv run python -m src.play --models CLAUDE3_SONNET GPT4 GPT35_TURBO

# Run game with the same model for all players
uv run python -m src.play --models CLAUDE3_SONNET

# Run simulator with a specific model
uv run python -m src.core.simulator -p 1 -o 1 -l 0 -m GPT4
```
